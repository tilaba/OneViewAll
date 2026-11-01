#!/bin/bash

# 基础路径和固定参数
YCBV_DIR="/root/data/YCBV"
EVAL_STRIDE=10

# 定义 21 个对象的参数: "ob_id video_idx frame_id"
# 数据直接提取自你的图片表格
params=(
    "1 0 0"
    "2 11 81"
    "3 6 1500"
    "4 2 1300"
    "5 2 400"
    "6 0 1000"
    "7 10 1000"
    "8 10 1000"
    "9 1 400"
    "10 2 400"
    "11 8 400"
    "12 6 81"
    "13 5 400"
    "14 0 400"
    "15 6 400"
    "16 7 900"    # 这是表格中高亮的 036_wood_block
    "17 3 100"
    "18 11 700"
    "19 0 1000"
    "20 0 100"
    "21 9 100"
)

# 循环遍历并执行
for param in "${params[@]}"; do
    read -r ob_id video_idx frame_id <<< "$param"
    
    save_dir="runs/probe_ycbv_model${ob_id}_best"
    
    echo "=================================================="
    echo "Running for Object ID: ${ob_id} (Video idx: ${video_idx}, Frame: ${frame_id})"
    echo "=================================================="
    
    python run_ycb_video.py \
        --ycbv_dir ${YCBV_DIR} \
        --ob_ids ${ob_id} \
        --ref_video_idx ${video_idx} \
        --ref_frame ${frame_id} \
        --save_ref_probe ${save_dir} \
        --eval_stride ${EVAL_STRIDE}
        
    # 如果你需要为第 16 类单独加上 crop_ratio，可以在这里加个 if 判断，例如：
    # if [ "$ob_id" -eq 16 ]; then
    #     python run_ycb_video.py ... --crop_ratio 1.3
    # fi
done

echo "所有 21 个对象运行完毕！"