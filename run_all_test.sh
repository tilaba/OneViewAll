#!/bin/bash

# 格式: "ob_id ref_frame ref_video_idx crop_ratio"
CONFIGS=(
    "5  400  02 1.1"
    "6  1000 00 1.1"
    "7  1000 10 1.1"
    "8  1000 10 1.1"
    "9  400  01 1.1"
    "10 400  02 1.1"
    "11 400  08 1.1"
    "12 100  06 1.1"
    "13 400  05 1.1"
    "14 400  00 1.1"
    "15 400  06 1.1"
    "16 900  07 1.3"
    "17 100  03 1.1"
    "18 700  11 1.1"
    "19 1000 00 1.1"
    "20 100  00 1.1"
    "21 100  09 1.1"
)

for cfg in "${CONFIGS[@]}"; do
    read -r ob_id ref_frame ref_video_idx crop_ratio <<< "$cfg"
    
    # 计算 probe_dir 名称（ob_id - 1，并补齐两位数字，例如 ob_id=1 对应 ob00）
    ob_minus_1=$((ob_id - 1))
    probe_dir=$(printf "runs/probe_ycbv_ob%02d_best" $ob_minus_1)

    echo "========================================="
    echo ">>> Running for ob_id: $ob_id (Video: $ref_video_idx, Frame: $ref_frame)"
    echo "========================================="

    # 执行你的命令
    python run_ycb_video.py \
        --ycbv_dir /root/data/YCBV \
        --ob_id $ob_id \
        --ref_frame $ref_frame \
        --ref_video_idx $ref_video_idx \
        --save_ref_probe $probe_dir \
        --eval_stride 1 \
        --crop_ratio $crop_ratio
        
    # 如果某次运行失败，停止后续运行并提示错误
    if [ $? -ne 0 ]; then
        echo "!!! Error occurred at ob_id $ob_id. Stopping execution."
        exit 1
    fi
done

echo "All evaluations completed successfully!"