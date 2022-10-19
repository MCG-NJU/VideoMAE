# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/ucf_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_0.75_videos_e3200'
# Set the path to UCF101 train set. 
DATA_PATH='YOUR_PATH/list_ucf/train.csv'


# batch_size can be adjusted according to number of GPUs
# this script is for 8 GPUs (1 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 \
         --master_port 12320 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.75 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --lr 3e-4 \
        --batch_size 24 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 3201 \
        --save_ckpt_freq 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}