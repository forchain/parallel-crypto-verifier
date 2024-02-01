#!/usr/bin/python3

import subprocess
import time

# 定义参数范围和步长
min_r = 0.1
max_r = 0.9
step = 0.1

# 存储不同参数取值的执行时间
execution_times = {}

# 循环测试不同的-r参数取值
for r in range(int(min_r * 10), int(max_r * 10) + 1, int(step * 10)):
    r_value = r / 10.0
    command = f"./verifier -x -r {r_value}"
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行./verifier程序
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 计算执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 存储执行时间
    execution_times[r_value] = execution_time


    # 打印执行时间
    print(f"-r参数值为 {r_value} 时的执行时间为 {execution_time} 秒")

# 找到最短执行时间对应的参数值
min_execution_time = min(execution_times.values())
best_r_value = [r for r, time in execution_times.items() if time == min_execution_time][0]

print(f"最短执行时间对应的-r参数值是：{best_r_value}")

