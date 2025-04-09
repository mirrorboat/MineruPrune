python -m llava.eval.eval_mixtral \
    --model-path ./playground/training/training_dirs/Qwen2-0.5B-Siglip-Pretrain-Qwen2-0.5B-Stage2-checkpoint-2-continue-box-1 \
    --data_dir ./playground/eval/omnidocbench_subset/images \
    --output_dir ./playground/eval/omnidocbench_subset/results_mixtral \
    --num-chunks 0 \
    --chunk-idx 0  \
    --conv-mode qwen_2