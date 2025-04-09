gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_got \
    --model-path /mnt/petrelfs/liuzheng/qwen2.5-siglip \
    --data_dir ./playground/eval/scene_bench_cn/images \
    --output_dir ./playground/eval/scene_bench_cn/results_3 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  \
    --conv-mode plain &
done

wait