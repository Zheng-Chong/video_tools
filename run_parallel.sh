#!/bin/bash

# 获取 GPU 数量 (使用 nvidia-smi 统计 L 字符行数)
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs."

# 如果你只想用特定的卡，可以手动修改这里，例如 NUM_GPUS=4
PYTHON_FILE="qwen/qwen_image_edit.py"

for ((i=0; i<$NUM_GPUS; i++))
do
    echo "Starting process on GPU $i..."
    # 使用 CUDA_VISIBLE_DEVICES 隔离环境，这样 Python 里的 cuda:0 就是真实的卡 i
    CUDA_VISIBLE_DEVICES=$i python $PYTHON_FILE --rank $i --world_size $NUM_GPUS &
done

# 等待所有后台进程结束
wait
echo "All parallel processes have completed."