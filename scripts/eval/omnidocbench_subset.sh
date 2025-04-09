gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_omnidocbench \
    --model-path ./playground/training/training_dirs/Qwen2-0.5B-Siglip-Pretrain-Qwen2-0.5B-Stage2-checkpoint-2-continue-box-1/checkpoint-144000 \
    --data_dir ./playground/eval/omnidocbench_subset/images \
    --output_dir ./playground/eval/omnidocbench_subset/results_test \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  \
    --conv-mode qwen_2 &
done

wait