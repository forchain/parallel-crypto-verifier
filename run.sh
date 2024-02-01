#!/bin/bash

# 默认值
max_sigs=1000000
increment=100000
current_sigs=100000
log_file="logs/verifier_log_$(date '+%Y%m%d%H%M%S').log"
run_times=1  # 新增参数，默认值为1
echo $log_file

while [ $# -gt 0 ]; do
    param="$1"
    case $param in
        -m)
            # 如果参数是"-m"，则将下一个参数的值赋给max_sigs，并跳过下一个参数
            max_sigs="$2"
            shift 2
            ;;
        -b)
            # 如果参数是"-b"，则将下一个参数的值赋给begin_sig，并跳过下一个参数
	    current_sigs="$2"
            shift 2
            ;;
        -i)
            # 如果参数是"-i"，则将下一个参数的值赋给increment，并跳过下一个参数
            increment="$2"
            shift 2
            ;;
        -n)
            # 如果参数是"-n"，则将下一个参数的值赋给run_times，并跳过下一个参数
            run_times="$2"
            shift 2
            ;;
        *)
            # 否则，将当前参数添加到filtered_params
            filtered_params="$filtered_params $param"
            shift
            ;;
    esac
done

echo  Max: $max_sigs From: $current_sigs Increment: $increment


# 循环调用程序，逐步增加签名数，并将输出同时写入文件和终端
while [ $current_sigs -le $max_sigs ]
do
	# 循环调用程序，逐步增加签名数，并将输出同时写入文件和终端
	for ((i=1; i<=$run_times; i++))
	do
    		echo "$current_sigs $i/$run_times"  # 显示当前运行次数

		# 调用程序并传递参数，并将结果追加到日志文件，同时实时显示在终端上
		./verifier -s $current_sigs  $filtered_params 2>&1 | tee -a "$log_file"
	done

    	# 增加签名数
    	current_sigs=$((current_sigs + increment))
done

