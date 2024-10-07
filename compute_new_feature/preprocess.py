import pandas as pd

# Load the CSV file
file_path = 'data_1.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Function to split sequences based on the tag column (assuming the tag column is labeled '标签')
# Function to split sequences based on the tag column (assuming the tag column is labeled '标签')
def split_signal_sequences(df):
    sequences = []
    current_sequence = []
    tag_count = 0  # To count occurrences of '0' or '1'
    
    for index, row in df.iterrows():
        if row['标签'] in ['0', '1']:  # Adjust '标签' if the column name is different
            tag_count += 1

            if tag_count == 2:
                # Append current sequence to sequences without including the current tag row
                sequences.append(pd.DataFrame(current_sequence))
                current_sequence = []  # Reset for the next sequence
                tag_count = 1  # Set tag_count to 1 to start the next sequence

        # Append the current row to the current sequence
        current_sequence.append(row)

    # Check if there's any remaining sequence
    if current_sequence:
        sequences.append(pd.DataFrame(current_sequence))

    return sequences

# Split the sequences
sequences = split_signal_sequences(data)


sequence_lengths = [len(seq) for seq in sequences]
from collections import Counter
# 统计每个长度出现的次数
length_counts = Counter(sequence_lengths)

# 打印每个长度的序列数量
for length, count in length_counts.items():
     print(f"长度为 {length} 的序列有 {count} 个")

# Find the longest and shortest sequences
max_length = max(sequence_lengths)
min_length = min(sequence_lengths)

# Print the results
print(f"最长的时间序列有 {max_length} 行数据")
print(f"最短的时间序列有 {min_length} 行数据")

# 将每个序列保存为单独的 CSV 文件
output_dir = 'sequences_output'  # 保存文件的目录
import os

# 如果目录不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存每个序列为 CSV 文件
for i, seq in enumerate(sequences):
    seq_file_name = f'{output_dir}/sequence_{i+1}.csv'
    seq.to_csv(seq_file_name, index=False)
    print(f"序列 {i+1} 保存为 {seq_file_name}")