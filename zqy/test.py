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

def compute_overall_accuracy(A, B, C, D):
    overall_accuracy = 0.35 * A + 0.3 * (1 - B) + 0.2 * (1 - C) + 0.15 * D
    return overall_accuracy

def truncated_seq(X_val, y_val, sequence_length_test):
    # 将测试集数据转换为5拍的序列
    test_data = []
    test_labels = []
    
    # 遍历 X_val 中的每个样本
    for i in range(X_val.shape[0]):
        for j in range(0, X_val.shape[1] - sequence_length_test + 1):  # 确保不越界
            sample = X_val[i, j:j + sequence_length_test, :]  # 输入部分，取5拍
            label = y_val[i]  # 标签部分（假设每个样本的标签是相同的）
            test_data.append(sample)
            test_labels.append(label)

    # 转换为 numpy 数组
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    return test_data, test_labels


# 交叉验证设置
kfold = KFold(n_splits=5, shuffle=True)

# 训练模型
def test_model(X, y, device):
    all_acc = []
    test_a, test_b, test_d = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f'Fold {fold + 1}')
        best_loss = float('inf')
        
        # 创建训练和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # X_val, y_val = truncated_seq(X_val, y_val, test_seq_len)
        
        ilen_train, ilen_val = init_length[train_idx], init_length[val_idx] # 原始样本的步长
        steps_train, steps_val = steps[train_idx], steps[val_idx] # 我们用的步长 全7数组
        
        C_train = (steps_train*y_train).sum() / ((ilen_train*y_train).sum() + 1e-8)
        C_val = (steps_val*y_val).sum() / ((ilen_val*y_val).sum() + 1e-8)
        print(f'C of val: {C_val*100:.2f}%')
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = LSTMModel(X_train.shape[2], hidden_size, output_size, num_layers).to(device)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        model.eval()
        
        model.load_state_dict(torch.load(best_model_path))
        
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

        test_a.append(A*100)
        test_b.append(B*100)
        test_d.append(D*100)
        all_acc.append(acc)
    
    avg_a, avg_b, avg_d = sum(test_a) / len(test_a), sum(test_b) / len(test_b), sum(test_d) / len(test_d)
    avg_acc = sum(all_acc) / len(all_acc)
    
    print(f"seq : {seq_len}. AVG: A = {avg_a:.2f}%, B = {avg_b:.2f}%, C = {C_val*100:.2f}%, D = {avg_d:.2f}%, overall acc = {avg_acc:.2f}%")


seq_len = 5

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

best_model_path = 'checkpoints/best_model.pth'

if __name__ == '__main__':
    # 训练模型
    test_model(X, y, device=device)
