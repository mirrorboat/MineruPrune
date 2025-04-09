gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_got \
    --model-path ./playground/training/training_dirs/Qwen2-0.5B-Siglip-Pretrain-checkpoint-4 \
    --data_dir ./playground/eval/scene_bench_en/images \
    --output_dir ./playground/eval/scene_bench_en/results \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  \
    --conv-mode plain &
done

wait