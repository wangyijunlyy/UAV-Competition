import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# 假设 'label' 列标记了每个样本的开始（即每个样本的第一个拍），空的表示拍属于上一个样本
def split_samples(df):
    samples = []
    # labels = []
    current_sample = []
    
    for i, row in df.iterrows():
        if row['label'] == -1:  # 属于上一个样本的拍
            current_sample.append(row.values)  # 忽略最后一列的标签
        else:
            if len(current_sample) > 0:
                samples.append(current_sample)

            current_sample = [row.values]  # 新样本的第一拍
    
    # 别忘了加入最后一个样本
    if len(current_sample) > 0:
        samples.append(current_sample)
    
    return samples

def pad_or_truncate_sequences(sequences, seq_length):
    padded_sequences = []
    init_length = []
    for seq in sequences:
        if len(seq) > seq_length:
            padded_sequences.append(np.array(seq[:seq_length], dtype=np.float32))  # 确保类型
            init_length.append(len(seq))
        else:
            # 如果长度不足，计算前面行的均值进行填充
            mean_value = np.mean(seq, axis=0)
            padded_seq = np.vstack([seq, np.tile(mean_value, (seq_length - len(seq), 1))])
            padded_sequences.append(np.array(padded_seq, dtype=np.float32))  # 填充到指定长度

            init_length.append(seq_length)
    return np.array(padded_sequences), np.array(init_length)
    
def prepare_data(seq_len):
    df = pd.read_csv('data.csv')
    df = df.drop(columns=['recorded_time'])

    # 确保标签列转换为数值，空值变为 NaN
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # 选择填充或删除 NaN 值
    df['label'] = df['label'].fillna(-1)  # 用 -1 填充 NaN 

    df.fillna(method='ffill', inplace=True) # forward fill,可以尝试不同的fill
    # 标准化其他特征
    scaler =  MinMaxScaler()
    features_to_scale = df.drop(columns=['label'])  # 保留标签列
    scaled_features = scaler.fit_transform(features_to_scale)
    # 创建新的 DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
    # 将标签列添加到新的 DataFrame 中
    scaled_df['label'] = df['label'].values.astype(int)  # 确保标签为整数类型
    
    samples = split_samples(scaled_df)
    # print(labels[0:20])
    # 计算每个样本的最大拍数（最大时间步长度）
    # seq_len = max([len(sample) for sample in samples]) # 11
    print('max num of data per sample: ', seq_len)
    padded_samples, init_length = pad_or_truncate_sequences(samples, seq_len)
    np.save('padded_sequences_7.npy', padded_samples)
    
    X = padded_samples[:, :, :-1]  # 特征数据
    y = padded_samples[:, 0, -1]  # 取最后一维作为标签，确保 y 的形状为 (num_samples,)

    return X, y, init_length
    
    # # 将数据集拆分为训练集和测试集
    # train_samples, test_samples, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    # nb_classes = len(np.unique(np.concatenate((train_labels, test_labels), axis=0)))
    # print(nb_classes)
    # print(f"训练集样本数: {train_samples.shape[0]}")
    # print(f"测试集样本数: {test_samples.shape[0]}")
    
    # return  train_samples, test_samples, train_labels, test_labels

def access_data():
    padded_samples =  np.load('/home/user/project/zqy/uav/padded_sequences_8.npy')
    
    X = padded_samples[:, :, :-1]  # 特征数据
    y = padded_samples[:, 0, -1]  # 取最后一维作为标签，确保 y 的形状为 (num_samples,)
    # print(y.shape)
    return X, y

if __name__ == '__main__':
    prepare_data()