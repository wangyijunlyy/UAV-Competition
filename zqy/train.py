import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import pickle
import random
import numpy as np
from data import prepare_data, access_data
from sklearn.model_selection import KFold

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 只使用第0和第1块显卡

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# 创建 PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # 确保标签为浮点型
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)
        # x: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch_size, seq_len, hidden_size]

        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])  # [batch_size, output_size]

        return out.view(-1)
    
# TCN模型
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_channels[1], output_size, kernel_size=1)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_size) => (batch_size, input_size, seq_len)
        out = self.tcn(x)      # 输出形状为 (batch_size, 1, seq_len)
        out = self.global_pool(out)  # 变为 (batch_size, 1, 1)
        return self.sigmoid(out.squeeze(1)).view(-1)  # 输出形状为 (batch_size, output_size)
    
def metrics(outputs, labels):
    predicted = (outputs > 0.5).float() # 阈值 0.5 将概率转换为二分类标签
    
    true_positives = (predicted * labels).sum()
    A = true_positives / (labels.sum() + 1e-8)
    
    false_positives = (predicted * (1 - labels)).sum()
    B = false_positives / ((1 - labels).sum() + 1e-8)

    # D: 无人机分类正确率 (Precision) 用正确率替代
    correct = (predicted == labels).sum().item()
    D = correct / labels.size(0)
    
    return A, B, D

def compute_overall_accuracy(A, B, C, D):
    overall_accuracy = 0.35 * A + 0.3 * (1 - B) + 0.2 * (1 - C) + 0.15 * D
    return overall_accuracy

# 交叉验证设置
kfold = KFold(n_splits=5, shuffle=True)

# 训练模型
def train_model(X, y, criterion, num_epochs, device):
    all_acc = []
    test_a, test_b, test_d = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f'Fold {fold + 1}')
        best_loss = float('inf')
        
        # 创建训练和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        ilen_train, ilen_val = init_length[train_idx], init_length[val_idx] # 原始样本的步长
        steps_train, steps_val = steps[train_idx], steps[val_idx] # 我们用的步长 全7数组
        
        C_train = (steps_train*y_train).sum() / ((ilen_train*y_train).sum() + 1e-8)
        C_val = (steps_val*y_val).sum() / ((ilen_val*y_val).sum() + 1e-8)
        print(f'C of train: {C_train*100:.2f}%, C of val: {C_val*100:.2f}%')
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = LSTMModel(X_train.shape[2], hidden_size, output_size, num_layers).to(device)
        # model = TCN(input_size=X_train.shape[2], output_size=1, num_channels=num_channels).to(device)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model.train()  # 设定为训练模式
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct, total = 0, 0
            overall_acc, true_positives, false_positives, labels_sum = 0.0, 0.0, 0.0, 0.0
            for samples, labels in train_loader:
                # 将数据加载到设备（CPU 或 GPU）
                samples, labels = samples.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(samples)
                loss = criterion(outputs, labels)  # 使用自定义损失
            
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算损失和准确率
                running_loss += loss.item()
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5).float() # 阈值 0.5 将概率转换为二分类标签
                total += labels.size(0)
                
                correct += (predicted == labels).sum().item()
                true_positives += (predicted * labels).sum().item()
                false_positives += (predicted * (1 - labels)).sum().item()
                labels_sum += labels.sum().item()

            # 记录训练损失
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            A = true_positives / labels_sum
            B = false_positives / (total - labels_sum)
            D = correct / total
            epoch_acc = compute_overall_accuracy(A, B, C_train, D)*100
            # epoch_acc = overall_acc / total * 100
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, A: {A*100:.2f}%, B: {B*100:.2f}%, D: {D*100:.2f}%, Overall_ACC: {epoch_acc:.2f}%')

            # 检查损失是否有改善
            if epoch_loss < best_loss and (epoch+1) % 10 == 0:
                best_loss = epoch_loss
                # 保存最佳模型
                torch.save(model.state_dict(), best_model_path)
                print(f'Best Model checkpoint saved at epoch {epoch+1} !!')

        model.load_state_dict(torch.load(best_model_path))
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        a, b, d, acc = test_model(model, test_loader, criterion, device, C_val)
        test_a.append(a)
        test_b.append(b)
        test_d.append(d)
        all_acc.append(acc)
    
    avg_a, avg_b, avg_d = sum(test_a) / len(test_a), sum(test_b) / len(test_b), sum(test_d) / len(test_d)
    avg_acc = sum(all_acc) / len(all_acc)
    
    print(f"seq : {seq_len}. AVG: A = {avg_a:.2f}%, B = {avg_b:.2f}%, C = {C_val*100:.2f}%, D = {avg_d:.2f}%, overall acc = {avg_acc:.2f}%")
    
# 测试模型
def test_model(model, test_loader, criterion, device, C_val):

    model.eval()  # 设定为测试模式
    correct, total = 0, 0
    overall_acc, true_positives, false_positives, labels_sum = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for samples, labels in test_loader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            outputs = torch.sigmoid(outputs)
            
            predicted = (outputs > 0.5).float() # 阈值 0.5 将概率转换为二分类标签
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_positives += (predicted * labels).sum().item()
            false_positives += (predicted * (1 - labels)).sum().item()
            labels_sum += labels.sum().item()
    
    A = true_positives / labels_sum
    B = false_positives / (total - labels_sum)
    D = correct / total
    acc = compute_overall_accuracy(A, B, C_val, D)*100
    print(f'TEST result -- A: {A*100:.2f}%, B: {B*100:.2f}%, C: {C_val*100:.2f}% D: {D*100:.2f}%, Overall_ACC: {acc:.2f}%')
    
    model.train()
    
    return A*100, B*100, D*100, acc

# 绘制损失曲线
def plot_loss_curve(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./train_loss.jpg')
    plt.show()

seq_len = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_samples, test_samples, train_labels, test_labels = prepare_data()
X, y, init_length = prepare_data(seq_len)

steps = []
for i in range(len(y)):
    steps.append(seq_len)
steps = np.array(steps)

# 模型初始化
# input_size = train_samples.shape[2]  # 输入特征维度
# print('shape of train: ', train_samples.shape)
hidden_size = 64  # LSTM隐层单元数
num_layers = 2  # LSTM层数
output_size = 1  # 输出大小（二分类）
# num_channels = [64, 32]  # TCN 中每层的通道数
# model = LSTMModel(input_size, hidden_size, output_size, num_layers=num_layers).to(device)
# model.load_state_dict(torch.load('checkpoints/best.pth', map_location=device))

# 训练参数
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# weights for pos
num_pos = 16974
num_neg = 41639
weight_positive = num_neg / num_pos
weight_negative = 1.0
# class_weights = torch.tensor([weight_negative, weight_positive], dtype=torch.float32)

# 损失函数和优化器
# criterion = nn.NLLLoss()
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_positive*3, dtype=torch.float32))
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()

best_model_path = 'checkpoints/best_model.pth'
train_losses = []

if __name__ == '__main__':
    # 训练模型
    train_model(X, y, criterion, num_epochs=num_epochs, device=device)
    # 绘制损失曲线
    plot_loss_curve(train_losses)