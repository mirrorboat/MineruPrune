export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export GPUS_PER_NODE=8
export NNODES=6
export MASTER_PORT=20955
# export MASTER_ADDR=SH-IDC1-10-140-24-22
export CPUS_PER_TASK=32

export SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export TRAINING_DATASET_JSON=$SCRIPT_DIR/../../data_recipe/box.yaml
# export TRAINING_DATASET_JSON=$SCRIPT_DIR/../../data_recipe/test.yaml

export TRAINING_NAME=Qwen2-0.5B-Siglip-0311-0405-anyres9-2

export NUM_TRAIN_EPOCHS=1

mkdir -p $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME

    # bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank 0 --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} $SCRIPT_DIR/../../llava/train/train_mem.py \
    # bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} $SCRIPT_DIR/../../llava/train/train_mem.py \

    # --quotatype=spot \

# srun -p mineru2_data \
#     --nodes=$NNODES \
#     --ntasks-per-node=1 \
#     --gres=gpu:$GPUS_PER_NODE \
#     --cpus-per-task=$CPUS_PER_TASK \
#     --kill-on-bad-exit=1 \
#     --debug \
    # -w SH-IDC1-10-140-24-81,SH-IDC1-10-140-24-24 \
    # -w SH-IDC1-10-140-24-100,SH-IDC1-10-140-24-95,SH-IDC1-10-140-24-89,SH-IDC1-10-140-24-83  \

srun -p mineru2_data \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:0 \
    --cpus-per-task=$CPUS_PER_TASK \
    --quotatype=spot \
    -w SH-IDC1-10-140-24-100,SH-IDC1-10-140-24-95,SH-IDC1-10-140-24-89,SH-IDC1-10-140-24-83,SH-IDC1-10-140-24-35,SH-IDC1-10-140-24-66  \
    bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} $SCRIPT_DIR/../../llava/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/../../scripts/zero3.json \
    --model_name_or_path /mnt/petrelfs/chenjingzhou/cjz/MyMinerU-LVLM/playground/training/training_dirs/Qwen2-0.5B-Siglip-0311 \
    --version qwen_2 \
    --data_path $TRAINING_DATASET_JSON \
    --vision_tower /mnt/petrelfs/chenjingzhou/cjz/opendatalab/cjz/ckpt/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(4x4)" \
    --mm_patch_merge_type spatial_unpad \
    --tune_l0_module_only True \
    --tune_entire_model True \
    --unfreeze_mm_vision_tower True \
    --save_vision_tower True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --mm_vision_tower_lr 2e-6 \
    --lag_lr 1.0 \
    --lagrangian_warmup_steps 500 \
    --weight_decay 0. \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 9500 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    2>&1 | tee $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME/training.log'


    # --tune_entire_model False \
    # --unfreeze_mm_vision_tower False \
    # --unfreeze_mm_projector True \
    # --freeze_mm_mlp_adapter False \

    # --tune_entire_model True \
    # --unfreeze_mm_vision_tower True \