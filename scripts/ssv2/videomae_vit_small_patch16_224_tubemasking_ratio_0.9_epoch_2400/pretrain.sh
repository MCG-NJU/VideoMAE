# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_small_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400'
# Set the path to SSV2 train set. 
DATA_PATH='YOUR_PATH/list_ssv2/train.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12320 --nnodes=8 --node_rank=$1 --master_addr=$2 \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_small_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 2401 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}