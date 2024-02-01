import re
import pandas as pd
from datetime import datetime

# 解析单条日志
def parse_log(log):
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2})")
    id_pattern = re.compile(r"\[ID:(\d+)\]")
    task_pattern = re.compile(r"第 (\d+) 号任务")
    duration_pattern = re.compile(r"耗费 (\d+\.\d+) 秒")

    time = time_pattern.search(log).group(1)
    node_id = int(id_pattern.search(log).group(1))
    task_num = int(task_pattern.search(log).group(1))
    duration = float(duration_pattern.search(log).group(1))

    task_count = 500000
    speed = task_count / duration

    return time, node_id, task_num, duration, speed

# 计算时间差
def calculate_time_difference(start_time, current_time):
    time_format = "%H:%M:%S"
    start_time = datetime.strptime(start_time, time_format)
    current_time = datetime.strptime(current_time, time_format)
    time_difference = current_time - start_time
    return time_difference.total_seconds()

# 读取并解析日志文件
def process_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if "向Master发送请求指令, 完成了第" in line]
    parsed_data = [parse_log(line) for line in filtered_lines]

    df = pd.DataFrame(parsed_data, columns=["time", "node", "task", "duration", "speed"])
    start_time = df.iloc[0]['time']
    df['elasped'] = df.apply(lambda row: calculate_time_difference(start_time, row['time']), axis=1)

    return df

# 示例：使用日志文件路径
log_file_path = 'logs/v-3080.100m.d.log'  # 替换为实际的日志文件路径
df = process_log_file(log_file_path)

# 保存为CSV文件
df.to_csv('parsed_log.csv', index=False)

