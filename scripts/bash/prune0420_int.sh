export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export GPUS_PER_NODE=8
export NNODES=1
export MASTER_PORT=20957
export CPUS_PER_TASK=32

export SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export TRAINING_DATASET_JSON=$SCRIPT_DIR/../../data_recipe/box.yaml
# export TRAINING_DATASET_JSON=$SCRIPT_DIR/../../data_recipe/evalset.yaml

export TRAINING_NAME=Qwen2-0.5B-Siglip-Pretrain-intermediate

export NUM_TRAIN_EPOCHS=1

mkdir -p $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME

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
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} $SCRIPT_DIR/../../llava/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/../../scripts/zero3.json \
    --model_name_or_path /mnt/petrelfs/chenjingzhou/cjz/MyMinerU-LVLM/playground/training/training_dirs/Qwen2-0.5B-Siglip-0311 \
    --version qwen_2 \
    --data_path $TRAINING_DATASET_JSON \
    --vision_tower /mnt/petrelfs/chenjingzhou/cjz/opendatalab/cjz/ckpt/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio square_anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(4x4)" \
    --mm_patch_merge_type spatial_unpad \
    --tune_l0_module_only True \
    --tune_entire_model True \
    --unfreeze_mm_vision_tower True \
    --save_vision_tower True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_box_start_end True \
    --bf16 True \
    --output_dir $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --mm_vision_tower_lr 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing False \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --lag_lr 0.2 \
    --lagrangian_warmup_steps 200 \
    --d_model 896 \
    --n_heads 14 \
    --intermediate_size 3684 \
    --n_kv_heads 2 \
    --n_layers 24 \
    --pruning_modules head hidden intermediate layer \
    2>&1 | tee $SCRIPT_DIR/../../playground/training/training_dirs/$TRAINING_NAME/training.log'


    # --eval_data_path /mnt/petrelfs/chenjingzhou/cjz/MyMinerU-LVLM/data_recipe/evalset.yaml \

    # --evaluation_strategy "steps" \
    # --eval_steps 50 \
