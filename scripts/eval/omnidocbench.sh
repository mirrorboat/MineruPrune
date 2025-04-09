gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_omnidocbench \
    --model-path ./playground/training/training_dirs/150M-Sam-Base-Pretrain-Qwen1.5-0.5B-Stage2-checkpoint-2 \
    --data_dir ./playground/eval/omnidocbench/images \
    --output_dir ./playground/eval/omnidocbench/results_1 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  \
    --conv-mode qwen_2 &
done

wait