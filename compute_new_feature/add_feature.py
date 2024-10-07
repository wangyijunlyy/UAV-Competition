# import pandas as pd
# import glob
# import os
# from tqdm import tqdm  # 导入 tqdm 库

# # 定义一个函数来计算瞬时径向加速度
# def calculate_instantaneous_acceleration(file_path, output_path):
#     # 加载 CSV 文件
#     df = pd.read_csv(file_path)
    
#     # 初始化加速度列，并将第一个点的加速度设为0
#     df['瞬时径向加速度(m/s²)'] = 0.0
    
#     # 计算瞬时径向加速度
#     for i in range(1, len(df)):
#         v1 = df['径向速率(m/s)'].iloc[i - 1]  # 前一个点的径向速率
#         v2 = df['径向速率(m/s)'].iloc[i]      # 当前点的径向速率
#         t = df['记录时间(s)'].iloc[i] - df['记录时间(s)'].iloc[i - 1]  # 当前点与前一个点的时间差
#         if t != 0:  # 避免除以零
#             df.loc[i, '瞬时径向加速度(m/s²)'] = (v2 - v1) / t  # 使用 .loc 进行赋值
            
#     # 保存结果到新的 CSV 文件
#     df.to_csv(output_path, index=False)

# # 遍历所有 CSV 文件
# all_files = glob.glob("/home/wsco/wyj/UAV/sequences_output/*.csv")  # 替换为实际路径

# # 处理每个文件，并添加进度条
# for file in tqdm(all_files, desc="处理文件", unit="文件"):
#     # 创建输出文件路径
#     output_file_name = os.path.basename(file).replace('.csv', '_acceleration.csv')
#     output_file_path = os.path.join("/home/wsco/wyj/UAV/acceleration_results", output_file_name)  # 替换为实际保存路径
#     calculate_instantaneous_acceleration(file, output_file_path)
import pandas as pd
import glob

# 定义文件夹路径
input_folder = "/home/wsco/wyj/UAV/acceleration_results/*.csv"  # 替换为实际路径
output_file = "/home/wsco/wyj/UAV/acceleration_results.csv"  # 替换为合并后保存的文件路径

# 获取所有 CSV 文件路径
all_files = glob.glob(input_folder)

# 按文件名排序，确保按顺序合并
all_files.sort()

# 读取并合并所有 CSV 文件
merged_df = pd.concat((pd.read_csv(file) for file in all_files), ignore_index=True)

# 保存合并后的结果到新的 CSV 文件
merged_df.to_csv(output_file, index=False)

print(f"合并完成，结果保存在：{output_file}")