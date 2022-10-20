# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_small_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/eval_lr_1e-3_epoch_40_layer_dacay_0.7'
# path to SSV2 annotation file (train.csv/val.csv/test.csv)
DATA_PATH='YOUR_PATH/list_ssv2'
# path to pretrain model
MODEL_PATH='YOUR_PATH/ssv2_videomae_pretrain_small_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/checkpoint-2399.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320 --nnodes=4  --node_rank=$1 --master_addr=$2 \
    run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 12 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 1e-3 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed 

