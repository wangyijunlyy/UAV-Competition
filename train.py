import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tsai.models.RNNAttention import LSTMAttention



class MyLSTMAttention(LSTMAttention):
    def __init__(self, c_in, c_out, seq_len, hidden_size=128, rnn_layers=1, bidirectional=False):
        super().__init__(c_in, c_out, seq_len, hidden_size, rnn_layers, bidirectional=bidirectional)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [bs x q_len x nvars]

        # RNN
        # x = x.transpose(2,1)    # [bs x nvars x q_len] --> [bs x q_len x nvars]
        output, _ = self.rnn(x) # output from all sequence steps: [bs x q_len x hidden_size * (1 + bidirectional)]

        # Attention Encoder
        z = self.encoder(output)                                             # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.sigmoid(self.head(z)).view(-1)      

# 加载 CSV 文件
file_path = 'data_1.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 删除记录时间列
df = df.drop(columns=['记录时间(s)'])

# 确保标签列转换为数值，空值变为 NaN
df['标签'] = pd.to_numeric(df['标签'], errors='coerce')

# 选择填充或删除 NaN 值
df['标签'] = df['标签'].fillna(-1)  # 用 -1 填充 NaN 

df.fillna(method='ffill', inplace=True)
#  检查其他列是否存在 NaN 或空值
nan_columns = df.drop(columns=['标签']).isnull().sum()

# 打印包含 NaN 值的列及其对应的 NaN 数量
nan_columns_with_values = nan_columns[nan_columns > 0]

if nan_columns_with_values.empty:
    print("其他列中没有 NaN 值")
else:
    print("以下列中存在 NaN 值：")
    print(nan_columns_with_values)
# 标准化其他特征
scaler =  MinMaxScaler()
features_to_scale = df.drop(columns=['标签'])  # 保留标签列
scaled_features = scaler.fit_transform(features_to_scale)

# 创建新的 DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns)

# 将标签列添加到新的 DataFrame 中
scaled_df['标签'] = df['标签'].values.astype(int)  # 确保标签为整数类型

# 根据标签进行分割
def split_signal_sequences(df):
    sequences = []
    current_sequence = []
    tag_count = 0  # 计数 '0' 或 '1' 的出现次数
    
    for index, row in df.iterrows():
        if row['标签'] in [0, 1]:  # 调整为数字比较
            tag_count += 1

            if tag_count == 2:
                # 将当前序列添加到 sequences 中，不包括当前标签行
                sequences.append(current_sequence)
                current_sequence = []  # 重置以开始下一个序列
                tag_count = 1  # 设置 tag_count 为 1 以开始下一个序列

        # 将当前行添加到当前序列
        current_sequence.append(row.values)  # 确保使用 row.values

    # 检查是否有剩余的序列
    if current_sequence:
        sequences.append(current_sequence)

    return sequences

# 分割序列
sequences = split_signal_sequences(scaled_df)

# 截取每个序列为长度 8，并进行填充
sequence_length = 8

def pad_or_truncate_sequences(sequences, seq_length):
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) > seq_length:
            # 截取多个子序列
            truncated_seqs = []
            for i in range(len(seq) - seq_length + 1):
                truncated_seqs.append(seq[i:i + seq_length])
            for ts in truncated_seqs:
                padded_sequences.append(np.array(ts, dtype=np.float32))  
            # padded_sequences.append(np.array(seq[:seq_length], dtype=np.float32))  
        else:
            # 如果长度不足，计算前面行的均值进行填充
            mean_value = np.mean(seq, axis=0)
            padded_seq = np.vstack([seq, np.tile(mean_value, (seq_length - len(seq), 1))])
            padded_sequences.append(np.array(padded_seq, dtype=np.float32))  # 填充到指定长度
            # # 如果长度不足 8，进行0填充
            # padded_seq = np.pad(seq, ((0, seq_length - len(seq)), (0, 0)), mode='constant', constant_values=0)
            # padded_sequences.append(np.array(padded_seq, dtype=np.float32))  # 确保类型
            
    return np.array(padded_sequences)

padded_sequences = pad_or_truncate_sequences(sequences, sequence_length)
# 保存 padded_sequences 到文件
np.save('padded_sequences_8_kuochong.npy', padded_sequences)
padded_sequences =  np.load('padded_sequences_8_kuochong.npy')




# # 加载 padded_sequences 从文件
# loaded_sequences = np.load('padded_sequences.npy')

# # 检查加载的数据
# print(loaded_sequences.shape)  # 输出形状确认加载成功
# 分割特征和标签
X = padded_sequences[:, :, :-1]  # 特征数据
y = padded_sequences[:, 0, -1]  # 取最后一维作为标签，确保 y 的形状为 (num_samples,)
print(y.shape)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
print(nb_classes)
# 创建 PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # 确保标签为浮点型
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 构建 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # 取最后一个时间步的输出
        
        
        return self.sigmoid(out).view(-1)

class Weight_LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Weight_LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        # 定义加权参数，对序列维度（8）进行加权
        self.weights = nn.Parameter(torch.randn(8))  # 对序列的每个时间步设置一个权重
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # shape: (batch_size, seq_length, hidden_size)

        # 对序列维度进行加权
        weighted_out = out * self.weights.unsqueeze(0).unsqueeze(2)  # shape: (batch_size, seq_length, hidden_size)

        # 对时间步进行加权求和
        weighted_sum = torch.sum(weighted_out, dim=1)  # shape: (batch_size, hidden_size)

        # 通过全连接层得到最终输出
        out = self.fc(weighted_sum)  # shape: (batch_size, output_size)

        return self.sigmoid(out).view(-1)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # 初始隐藏状态
        out, _ = self.gru(x, h0)  # 前向传播计算输出和隐藏状态
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为分类结果
        return self.sigmoid(out).view(-1)

# 模型超参数
print(X_train.shape)
input_size = X_train.shape[2]  # 特征数量
seq_len = X_train.shape[1]
hidden_size = 64  # LSTM 隐藏层大小
output_size = 1   # 二分类任务的输出
num_layers = 3   # LSTM 层数
learning_rate = 0.001

# 实例化模型
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
# model = GRUModel(input_size, hidden_size, output_size, num_layers)
# model = MyLSTMAttention(c_in=input_size, c_out=output_size, seq_len=seq_len, rnn_layers=num_layers)
# 使用 CrossEntropyLoss 损失函数
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练 LSTM 模型
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 300
# 初始化最佳准确率
# 记录损失的列表
train_losses = []
# 用于判断损失是否增加的计数器
no_improvement_count = 0
# 最佳损失初始化为无穷大
best_loss = float('inf')
best_train_accuracy = 0.0
best_model_path = 'best_model.pth'
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # print(y_batch)
        # 前向传播
        outputs = model(X_batch)
     
        loss = criterion(outputs, y_batch)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算训练准确率
        predicted = (outputs > 0.5).float()  # 阈值 0.5 将概率转换为二分类标签
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = correct_train / total_train * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
    # # 保存最佳模型
    # if train_accuracy > best_train_accuracy:
    #     best_train_accuracy = train_accuracy
    #     torch.save(model.state_dict(), best_model_path)
    #     print(f'Saved best model with accuracy: {best_train_accuracy:.2f}%')

    # 检查损失是否有改善
    if train_loss < best_loss:
        best_loss = train_loss
        no_improvement_count = 0
        # 保存最佳模型
        torch.save(model.state_dict(), best_model_path)
    else:
        no_improvement_count += 1
    
    # 如果5个epoch没有改善，则停止训练
    if no_improvement_count >= 5:
        print("Loss has not improved for 5 epochs. Stopping training.")
        break

# 绘制损失图
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('loss.png')
# 测试模型
# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs.view(-1), y_batch.view(-1))  # 这里调整了形状
        test_loss += loss.item()
        # 计算测试准确率
        predicted = (outputs > 0.5).float()  # 阈值 0.5
        correct_test += (predicted == y_batch).sum().item()
        total_test += y_batch.size(0)

    test_accuracy = correct_test / total_test * 100
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')
 