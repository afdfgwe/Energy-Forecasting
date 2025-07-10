import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.checkpoint import checkpoint
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import os


# --- 模型定义 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device="cpu", max_len=1000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).type(torch.float32)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = self.pe.unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PreLayer(nn.Module):
    def __init__(self, d_model, in_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)

    def forward(self, x):
        return self.linear(x)


class PostLayer(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.query(x).view(B, N, self.n_head, self.d_k).transpose(1, 2)
        k = self.key(x).view(B, N, self.n_head, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, N, self.n_head, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == float('-inf'), float('-inf'))
        attn = self.softmax(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, memory, mask=None):
        B, N, D = x.shape
        B_mem, N_mem, D_mem = memory.shape

        q = self.query(x).view(B, N, self.n_head, self.d_k).transpose(1, 2)
        k = self.key(memory).view(B_mem, N_mem, self.n_head, self.d_k).transpose(1, 2)
        v = self.value(memory).view(B_mem, N_mem, self.n_head, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == float('-inf'), float('-inf'))
        attn = self.softmax(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_hidnum):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_hidnum)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_hidnum, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, n_head)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.attn(x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn1 = SelfAttention(d_model, n_head)
        self.attn2 = CrossAttention(d_model, n_head)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout1(self.attn1(x, mask=tgt_mask)))
        x = self.norm2(x + self.dropout2(self.attn2(x, memory, mask=src_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, N, d_model, n_head, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, ff_hidnum, drop_out) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, N, d_model, n_head, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, ff_hidnum, drop_out) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, device, d_model, in_dim_enc, in_dim_dec, N_enc, N_dec, h_enc, h_dec,
                 ff_hidnum, dropout_model, use_checkpoint=False):
        super().__init__()
        self.device = device
        self.use_checkpoint = use_checkpoint

        self.x_pre = PreLayer(d_model, in_dim_enc)
        self.y_pre = PreLayer(d_model, in_dim_dec)
        self.pos_enc = PositionalEncoding(d_model, device=device, max_len=1000)

        self.enc = Encoder(N_enc, d_model, h_enc, ff_hidnum, dropout_model)
        self.dec = Decoder(N_dec, d_model, h_dec, ff_hidnum, dropout_model)
        self.post = PostLayer(d_model, 1)

    def make_tgt_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask.to(self.device)

    def forward(self, x, y_combined):
        tgt_mask = self.make_tgt_mask(y_combined.size(1))
        x_emb = self.pos_enc(self.x_pre(x))
        y_emb = self.pos_enc(self.y_pre(y_combined))

        if self.use_checkpoint and self.training:
            memory = checkpoint(self.enc, x_emb, use_reentrant=False)
            out = checkpoint(self.dec, y_emb, memory, None, tgt_mask, use_reentrant=False)
        else:
            memory = self.enc(x_emb)
            out = self.dec(y_emb, memory, tgt_mask=tgt_mask)

        out = self.post(out)
        out = out.squeeze(-1)
        return out

    def predict(self, x, pred_len, future_time_features):
        self.eval()
        x_emb = self.pos_enc(self.x_pre(x))
        memory = self.enc(x_emb)

        # Start with a zero token for the decoder
        output_sequence = torch.zeros(x.size(0), 1, 1).to(self.device)

        with torch.no_grad():
            for i in range(pred_len):
                # Use time features for all previously generated tokens
                time_features_so_far = future_time_features[:, :output_sequence.size(1), :]
                decoder_input_combined = torch.cat([output_sequence, time_features_so_far], dim=-1)

                tgt_mask = self.make_tgt_mask(decoder_input_combined.size(1))
                y_emb = self.pos_enc(self.y_pre(decoder_input_combined))
                out = self.dec(y_emb, memory, tgt_mask=tgt_mask)

                # Get prediction for the last token only
                next_token_pred = self.post(out[:, -1:, :])

                # Append the new prediction to the sequence
                output_sequence = torch.cat([output_sequence, next_token_pred], dim=1)

        # Return sequence excluding the initial zero token
        return output_sequence[:, 1:, :].squeeze(-1)


# --- 数据处理 ---
def preprocess_data():
    print("Loading and preprocessing data...")
    try:
        train_df_raw = pd.read_csv('train.csv', na_values=['?'])
        test_df_raw = pd.read_csv('test.csv', header=None, na_values=['?'])
        test_df_raw.columns = train_df_raw.columns
    except FileNotFoundError:
        print("Error: 'train.csv' or 'test.csv' not found. Please place them in the same directory.")
        return None, None, None, None, None

    test_start_date = pd.to_datetime(test_df_raw.iloc[0, 0]).date()

    df = pd.concat([train_df_raw, test_df_raw], ignore_index=True)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.ffill(inplace=True)

    df['Sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df[
        'Sub_metering_2'] - df['Sub_metering_3']

    agg_rules = {
        'Global_active_power': 'sum', 'Global_reactive_power': 'sum',
        'Voltage': 'mean', 'Global_intensity': 'mean',
        'Sub_metering_1': 'sum', 'Sub_metering_2': 'sum', 'Sub_metering_3': 'sum',
        'Sub_metering_remainder': 'sum', 'RR': 'first', 'NBJRR1': 'first',
        'NBJRR5': 'first', 'NBJRR10': 'first', 'NBJBROU': 'first'
    }
    daily_df = df.resample('D').agg(agg_rules)
    daily_df.ffill(inplace=True)

    # 创建基于时间的特征
    daily_df['day_of_week'] = daily_df.index.dayofweek
    daily_df['day_of_year'] = daily_df.index.dayofyear
    daily_df['month'] = daily_df.index.month
    daily_df['year'] = daily_df.index.year - daily_df.index.year.min()

    # 对周期性特征进行编码
    for feat, max_val in [('day_of_week', 7), ('day_of_year', 366), ('month', 12)]:
        daily_df[f'{feat}_sin'] = np.sin(2 * np.pi * daily_df[feat] / max_val)
        daily_df[f'{feat}_cos'] = np.cos(2 * np.pi * daily_df[feat] / max_val)
    daily_df.drop(['day_of_week', 'day_of_year', 'month'], axis=1, inplace=True)

    time_feature_cols = ['year', 'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
                         'month_sin', 'month_cos']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_df)
    scaled_df = pd.DataFrame(scaled_data, columns=daily_df.columns, index=daily_df.index)

    train_data = scaled_df[scaled_df.index.date < test_start_date]
    test_data = scaled_df[scaled_df.index.date >= test_start_date]

    print("Data preprocessing complete.")
    return train_data, test_data, scaler, daily_df.columns, time_feature_cols


class PowerDataset(Dataset):
    def __init__(self, data, input_len, output_len, feature_cols, target_col, time_feature_cols):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.time_feature_cols = time_feature_cols

    def __len__(self):
        return len(self.data) - self.input_len - self.output_len + 1

    def __getitem__(self, idx):
        input_start = idx
        input_end = idx + self.input_len
        output_end = input_end + self.output_len

        encoder_input = self.data[self.feature_cols].iloc[input_start:input_end].values

        target_sequence = self.data[self.target_col].iloc[input_end:output_end].values
        decoder_time_features = self.data[self.time_feature_cols].iloc[input_end:output_end].values

        # 教师强制的解码器输入：滞后的目标序列
        decoder_input = np.zeros_like(target_sequence)
        if len(target_sequence) > 1:
            decoder_input[1:] = target_sequence[:-1]

        return (torch.FloatTensor(encoder_input),
                torch.FloatTensor(decoder_input).unsqueeze(-1),
                torch.FloatTensor(decoder_time_features),
                torch.FloatTensor(target_sequence))


# --- 训练与评估 ---
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggerd after {self.patience} epochs of no improvement.")
                self.early_stop = True


def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for enc_in, dec_in, dec_time, dec_out in dataloader:
        enc_in, dec_out = enc_in.to(device), dec_out.to(device)
        # 结合滞后目标和未来时间特征进行解码器输入
        dec_in_combined = torch.cat([dec_in, dec_time], dim=-1).to(device)

        optimizer.zero_grad()
        output = model(enc_in, dec_in_combined)
        loss = criterion(output, dec_out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validation_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for enc_in, dec_in, dec_time, dec_out in dataloader:
            enc_in, dec_out = enc_in.to(device), dec_out.to(device)
            dec_in_combined = torch.cat([dec_in, dec_time], dim=-1).to(device)
            output = model(enc_in, dec_in_combined)
            loss = criterion(output, dec_out)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_model(model, dataloader, scaler, device, target_col_idx, pred_len):
    model.eval()
    total_mse = 0
    total_mae = 0

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    plot_preds_inv = None
    plot_trues_inv = None

    with torch.no_grad():
        for i, (enc_in, _, dec_time, dec_out_true_scaled) in enumerate(dataloader):
            enc_in, dec_time = enc_in.to(device), dec_time.to(device)

            pred_out_scaled = model.predict(enc_in, pred_len, dec_time)

            pred_out_reshaped = pred_out_scaled.cpu().numpy()
            dec_out_true_reshaped = dec_out_true_scaled.numpy()

            preds_inv_list = []
            trues_inv_list = []
            for j in range(pred_out_reshaped.shape[0]):
                dummy_pred = np.zeros((pred_out_reshaped.shape[1], scaler.n_features_in_))
                dummy_true = np.zeros((dec_out_true_reshaped.shape[1], scaler.n_features_in_))

                dummy_pred[:, target_col_idx] = pred_out_reshaped[j, :]
                dummy_true[:, target_col_idx] = dec_out_true_reshaped[j, :]

                preds_inv_list.append(scaler.inverse_transform(dummy_pred)[:, target_col_idx])
                trues_inv_list.append(scaler.inverse_transform(dummy_true)[:, target_col_idx])

            preds_inv = torch.FloatTensor(np.array(preds_inv_list)).to(device)
            trues_inv = torch.FloatTensor(np.array(trues_inv_list)).to(device)

            if i == 0:
                plot_preds_inv = preds_inv.cpu().numpy()[0]
                plot_trues_inv = trues_inv.cpu().numpy()[0]

            total_mse += criterion_mse(preds_inv, trues_inv).item()
            total_mae += criterion_mae(preds_inv, trues_inv).item()

    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)

    return avg_mse, avg_mae, plot_preds_inv, plot_trues_inv


# --- 实验流程 ---
def run_experiment(pred_len, train_data, test_data, scaler, feature_cols, target_col, time_feature_cols):
    print(f"\n----- Starting Experiment: Predicting {pred_len} days -----")

    # 参数配置
    INPUT_LEN = 90
    NUM_EPOCHS = 1000
    PATIENCE = 40
    NUM_RUNS = 5
    BATCH_SIZE = 32
    LR = 5e-5
    D_MODEL = 64
    N_HEAD = 4
    N_LAYERS = 2
    FF_HIDNUM = 128
    DROPOUT = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建数据集
    full_train_dataset = PowerDataset(train_data, INPUT_LEN, pred_len, feature_cols, target_col, time_feature_cols)
    test_data_with_history = pd.concat([train_data.tail(INPUT_LEN), test_data])
    test_dataset = PowerDataset(test_data_with_history, INPUT_LEN, pred_len, feature_cols, target_col,
                                time_feature_cols)

    # 拆分训练数据进行验证
    val_size = int(len(full_train_dataset) * 0.2)
    train_size = len(full_train_dataset) - val_size
    train_subset = Subset(full_train_dataset, range(train_size))
    val_subset = Subset(full_train_dataset, range(train_size, len(full_train_dataset)))

    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    target_col_idx = list(feature_cols).index(target_col)

    # 解码器输入维度= 1 (对于滞后值) +时间特征数
    in_dim_dec = 1 + len(time_feature_cols)

    all_mse = []
    all_mae = []
    last_run_preds, last_run_trues = None, None

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")

        checkpoint_path = f'checkpoint_run{run}_len{pred_len}.pt'
        early_stopper = EarlyStopper(patience=PATIENCE, path=checkpoint_path)

        model = Transformer(
            device=device, in_dim_enc=len(feature_cols), in_dim_dec=in_dim_dec,
            d_model=D_MODEL, N_enc=N_LAYERS, N_dec=N_LAYERS, h_enc=N_HEAD,
            h_dec=N_HEAD, ff_hidnum=FF_HIDNUM, dropout_model=DROPOUT
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            train_loss = train_loop(model, train_loader, optimizer, criterion, device)
            val_loss = validation_loop(model, val_loader, criterion, device)
            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s")

            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                break

        print("Loading best model from checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path))

        eval_start_time = time.time()
        mse, mae, last_run_preds, last_run_trues = eval_model(model, test_loader, scaler, device, target_col_idx,
                                                              pred_len)
        eval_time = time.time() - eval_start_time
        print(f"Run {run + 1} Evaluation | MSE: {mse:.2f}, MAE: {mae:.2f} | Time: {eval_time:.2f}s")
        all_mse.append(mse)
        all_mae.append(mae)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    print("\n----- Experiment Summary -----")
    print(f"Prediction Length: {pred_len} days")
    print(f"Average MSE over {NUM_RUNS} runs: {np.mean(all_mse):.2f} (std: {np.std(all_mse):.2f})")
    print(f"Average MAE over {NUM_RUNS} runs: {np.mean(all_mae):.2f} (std: {np.std(all_mae):.2f})")

    if last_run_preds is not None and last_run_trues is not None:
        plt.figure(figsize=(15, 7))
        plt.plot(last_run_trues, label='Ground Truth')
        plt.plot(last_run_preds, label='Prediction', linestyle='--')
        plt.title(f'Power Consumption Prediction ({pred_len} Days)')
        plt.xlabel('Days')
        plt.ylabel('Global Active Power (Sum per Day)')
        plt.legend()
        plt.grid(True)
        plot_filename = f'prediction_{pred_len}_days.png'
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.close()


if __name__ == '__main__':
    train_data, test_data, scaler, all_cols, time_cols = preprocess_data()

    if train_data is not None:
        FEATURE_COLS = all_cols
        TARGET_COL = 'Global_active_power'

        run_experiment(pred_len=90, train_data=train_data, test_data=test_data,scaler=scaler, feature_cols=FEATURE_COLS, target_col=TARGET_COL, time_feature_cols=time_cols)

        run_experiment(pred_len=365, train_data=train_data, test_data=test_data,
                       scaler=scaler, feature_cols=FEATURE_COLS, target_col=TARGET_COL, time_feature_cols=time_cols)

