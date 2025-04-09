gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=8

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_omnidocbench \
    --model-path /mnt/petrelfs/liuzheng/Mineru2/Mineru_VLM-Dev/playground/training/training_dirs/Qwen2-0.5B-Siglip-Pretrain-Qwen2-0.5B-Stage2-checkpoint-2-continue-exam_table \
    --data_dir ./playground/eval/omnidocbench/images \
    --output_dir ./playground/eval/omnidocbench/results_table_1 \
    --num-chunks 32 \
    --chunk-idx $((IDX+8))  \
    --conv-mode qwen_2 &
done

wait