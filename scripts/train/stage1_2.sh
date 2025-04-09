export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export GPUS_PER_NODE=8
export NNODES=4
export MASTER_PORT=29945
export CPUS_PER_TASK=32

export SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export TRAINING_DATASET_JSON=$SCRIPT_DIR/../../data_recipe/stage1.yaml

export TRAINING_NAME=Opt-150M-Sam-Base-Pretrain-checkpoint-2

export NUM_TRAIN_EPOCHS=3

mkdir -p $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME

srun -p mineru2 \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --job-name stage1_opt \
    bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} $SCRIPT_DIR/../../llava/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/../../scripts/zero3.json \
    --model_name_or_path $SCRIPT_DIR/../../playground/model/opt-125m \
    --version plain \
    --data_path $TRAINING_DATASET_JSON \
    --vision_tower $SCRIPT_DIR/../../playground/model/sam-vit-base \
    --mm_projector_type mlp2x_gelu \
    --tune_entire_model True \
    --unfreeze_mm_vision_tower True \
    --save_vision_tower True \
    --group_by_modality_length True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    2>&1 | tee $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME/training.log'