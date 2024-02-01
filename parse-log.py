import re
import csv
from collections import defaultdict

def parse_log_and_generate_csv(log_file_path, csv_file_path):
    # 更新正则表达式以匹配特殊字符
    pattern = r"\[([\w\+\-]+)\] (\d+) verify took (\d+\.\d+) seconds to execute"
    data = defaultdict(lambda: defaultdict(list))

    # 读取日志文件
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                type_, count, seconds = match.groups()
                data[type_][int(count)].append(float(seconds))  # 将count转换为整数

    # 计算平均值
    averages = {}
    for type_ in data:
        averages[type_] = {count: "{:.6f}".format(sum(times)/len(times)) for count, times in data[type_].items()}

    # 按数字排序列名
    counts = sorted(set(count for type_ in averages for count in averages[type_]))

    # 写入CSV文件
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['TYPE'] + counts)
        for type_ in sorted(averages.keys()):
            row = [type_] + [averages[type_].get(count, '') for count in counts]
            csvwriter.writerow(row)

# 示例用法
parse_log_and_generate_csv('log.txt', 'output.csv')

