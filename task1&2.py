import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random
import math
import os
import matplotlib.pyplot as plt

# --- 1. 配置参数 ---
class Config:
    train_path = 'train.csv'
    test_path = 'test.csv'
    plot_dir = 'plots'

    input_len = 90
    pred_len_short = 90
    pred_len_long = 365

    n_experiments = 5
    epochs = 50
    batch_size = 32
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42


# 设置随机种子以保证可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(Config.seed)
print(f"Using device: {Config.device}")


# --- 2. 数据处理 ---
def load_and_process_data(cfg):
    """加载、合并、预处理数据并按天重采样"""
    try:
        train_df = pd.read_csv(cfg.train_path, sep=',', low_memory=False)
        test_df = pd.read_csv(cfg.test_path, sep=',', header=None, low_memory=False)
        test_df.columns = train_df.columns
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure train.csv and test.csv are in the same directory.")
        return None

    df = pd.concat([train_df, test_df], ignore_index=True)

    # 处理时间格式
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')

    # 处理缺失值 '?'
    df.replace('?', np.nan, inplace=True)
    df.dropna(subset=['Global_active_power'], inplace=True)  # 关键目标列的缺失直接丢弃

    numeric_cols = df.columns.drop('DateTime')
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 对所有列进行前向填充
    df.ffill(inplace=True)

    # 特征工程
    df['sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - \
                                   (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])

    df.set_index('DateTime', inplace=True)

    # 按天重采样
    agg_rules = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'sub_metering_remainder': 'sum',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }

    daily_df = df.resample('D').agg(agg_rules)
    # 填充因重采样产生的缺失天
    daily_df.ffill(inplace=True)

    return daily_df


class PowerDataset(Dataset):
    """
    创建PyTorch数据集。
    一个样本(Sample)由一个input序列和一个紧随其后的output序列构成。
    """

    def __init__(self, data, input_len, pred_len):
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len
        self.target_col_idx = 0  # 目标变量 'Global_active_power'

    def __len__(self):
        # 总样本数由滑动窗口决定，窗口大小为 input_len + pred_len
        return len(self.data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        # input_seq 和 target_seq 是连续的
        input_seq = self.data[idx: idx + self.input_len]
        target_seq = self.data[idx + self.input_len: idx + self.input_len + self.pred_len, self.target_col_idx]

        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)


# --- 3. 模型定义 ---

class LSTMModel(nn.Module):
    def __init__(self, input_features, hidden_dim, num_layers, output_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.linear(last_time_step_out)
        return prediction


class TransformerModel(nn.Module):
    def __init__(self, input_features, d_model, nhead, d_hid, nlayers, output_len, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_features, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, output_len)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output


# --- 4. 训练、评估与绘图 ---
def train_model(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, criterion, target_scaler, device):
    model.eval()
    total_loss_mse = 0
    total_loss_mae = 0
    mae_criterion = nn.L1Loss()

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            # 为了反归一化，需要将tensor整形为scaler接受的2D array
            y_reshaped = y.cpu().numpy().reshape(-1, 1)
            output_reshaped = output.cpu().numpy().reshape(-1, 1)

            # 反归一化以计算真实尺度的误差
            y_inv_flat = target_scaler.inverse_transform(y_reshaped)
            output_inv_flat = target_scaler.inverse_transform(output_reshaped)

            # 整形回原来的批次维度以计算批次损失
            y_inv = y_inv_flat.reshape(y.shape)
            output_inv = output_inv_flat.reshape(output.shape)

            loss_mse = criterion(torch.tensor(output_inv), torch.tensor(y_inv))
            loss_mae = mae_criterion(torch.tensor(output_inv), torch.tensor(y_inv))

            total_loss_mse += loss_mse.item()
            total_loss_mae += loss_mae.item()

    return total_loss_mse / len(data_loader), total_loss_mae / len(data_loader)


def plot_and_save(model, data_loader, target_scaler, device, model_name, pred_len, plot_dir):
    model.eval()
    with torch.no_grad():
        # 取测试集的第一个batch来进行可视化
        x, y = next(iter(data_loader))
        x, y = x.to(device), y.to(device)

        output = model(x)

        # 从batch中取第一个样本进行绘图
        truth_sample = y[0].cpu().numpy().reshape(-1, 1)
        prediction_sample = output[0].cpu().numpy().reshape(-1, 1)

        # 反归一化
        truth_inv = target_scaler.inverse_transform(truth_sample).flatten()
        prediction_inv = target_scaler.inverse_transform(prediction_sample).flatten()

        plt.figure(figsize=(15, 7))
        plt.plot(truth_inv, label='Ground Truth', color='blue', linewidth=2)
        plt.plot(prediction_inv, label='Prediction', color='red', linestyle='--', linewidth=2)
        plt.title(f'Prediction vs. Ground Truth for {model_name} ({pred_len} Days)', fontsize=16)
        plt.xlabel('Time (Days into Prediction Horizon)', fontsize=12)
        plt.ylabel('Daily Global Active Power (summed Wh)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 保存图片
        plot_path = os.path.join(plot_dir, f'{model_name}_{pred_len}_days_prediction.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")


# --- 5. 主执行逻辑 ---
def main():
    cfg = Config()
    # 自动创建保存图片的文件夹
    os.makedirs(cfg.plot_dir, exist_ok=True)

    print("Loading and processing data...")
    daily_df = load_and_process_data(cfg)
    if daily_df is None: return

    # 将目标变量放在第一列，方便数据集和评估函数处理
    cols = ['Global_active_power'] + [col for col in daily_df.columns if col != 'Global_active_power']
    daily_df = daily_df[cols]

    # 按日期分割训练集和测试集
    split_date = '2009-01-01'
    train_data = daily_df[daily_df.index < split_date]

    # 数据归一化（非常重要：只在训练集上fit，然后transform所有数据）
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    full_data_scaled = scaler.transform(daily_df)

    # 创建一个只用于目标值的scaler，方便反归一化
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_data[['Global_active_power']])

    scenarios = [
        ("LSTM", cfg.pred_len_short),
        ("LSTM", cfg.pred_len_long),
        ("Transformer", cfg.pred_len_short),
        ("Transformer", cfg.pred_len_long)
    ]

    for model_name, pred_len in scenarios:
        print("\n" + "=" * 50)
        print(f"Running scenario: Model={model_name}, Prediction Length={pred_len} days")
        print("=" * 50)

        # 创建训练数据集
        train_dataset = PowerDataset(train_data_scaled, cfg.input_len, pred_len)

        # 创建测试数据集，需要包含训练集末尾的数据以构建输入序列
        test_start_index_in_full_data = len(train_data) - cfg.input_len
        test_data_for_sequences = full_data_scaled[test_start_index_in_full_data:]
        test_dataset = PowerDataset(test_data_for_sequences, cfg.input_len, pred_len)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        all_mse = []
        all_mae = []

        final_model = None  # 用于保存最后一轮实验的模型以供绘图
        for i in range(cfg.n_experiments):
            print(f"\n--- Experiment {i + 1}/{cfg.n_experiments} ---")

            # 初始化模型
            if model_name == "LSTM":
                model = LSTMModel(input_features=daily_df.shape[1], hidden_dim=64, num_layers=2, output_len=pred_len)
            else:
                model = TransformerModel(input_features=daily_df.shape[1], d_model=64, nhead=4, d_hid=128, nlayers=2,
                                         output_len=pred_len)

            model.to(cfg.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

            # 训练模型
            for epoch in tqdm(range(cfg.epochs), desc=f"Training Exp {i + 1}"):
                train_loss = train_model(model, train_loader, criterion, optimizer, cfg.device)
                if (epoch + 1) % 10 == 0:
                    # 使用tqdm.write防止破坏进度条，并使用局部变量
                    tqdm.write(f"  Exp {i + 1}, Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.4f}")

            # 评估模型
            mse, mae = evaluate_model(model, test_loader, criterion, target_scaler, cfg.device)
            print(f"Experiment {i + 1} Results -> MSE: {mse:.2f}, MAE: {mae:.2f}")
            all_mse.append(mse)
            all_mae.append(mae)

            if i == cfg.n_experiments - 1:
                final_model = model  # 保存最后一个模型

        # 平均结果
        print("\n" + "-" * 20 + " Final Results " + "-" * 20)
        print(f"Scenario: Model={model_name}, Prediction Length={pred_len}")
        print(f"Average MSE over {cfg.n_experiments} runs: {np.mean(all_mse):.2f} (std: {np.std(all_mse):.2f})")
        print(f"Average MAE over {cfg.n_experiments} runs: {np.mean(all_mae):.2f} (std: {np.std(all_mae):.2f})")

        # 使用最后一轮的模型进行绘图和保存
        print("\nGenerating prediction plot for the last experiment's model...")
        plot_and_save(final_model, test_loader, target_scaler, cfg.device, model_name, pred_len, cfg.plot_dir)


if __name__ == '__main__':
    main()
