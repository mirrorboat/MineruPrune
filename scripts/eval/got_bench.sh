gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_got \
    --model-path ./playground/training/training_dirs/Opt-150M-Clip-Pretrain-checkpoint-1 \
    --data_dir ./playground/eval/got_format_benchmark/images \
    --output_dir ./playground/eval/got_format_benchmark/results_2 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  \
    --conv-mode plain &
done

wait