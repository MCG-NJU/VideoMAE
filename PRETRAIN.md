# Pre-training VideoMAE 

## Original Implementation

The implementation of our VideoMAE supports **multi-node distributed training**. We provide the **off-the-shelf** scripts in the [scripts folder](scripts).

-  For example, to pre-train VideoMAE ViT-Base on **Something-Something V2** with 64 GPUs (8 nodes x 8 GPUs), you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800'
  DATA_PATH='YOUR_PATH/list_ssv2/train.csv'
  
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
          --master_port 12320 --nnodes=8 \
          --node_rank=0 --master_addr=$ip_node_0 \
          run_mae_pretraining.py \
          --data_path ${DATA_PATH} \
          --mask_type tube \
          --mask_ratio 0.9 \
          --model pretrain_videomae_base_patch16_224 \
          --decoder_depth 4 \
          --batch_size 32 \
          --num_frames 16 \
          --sampling_rate 2 \
          --opt adamw \
          --opt_betas 0.9 0.95 \
          --warmup_epochs 40 \
          --save_ckpt_freq 20 \
          --epochs 801 \
          --log_dir ${OUTPUT_DIR} \
          --output_dir ${OUTPUT_DIR}
  ```

  on the first node. On other nodes, run the same command with `--node_rank 1`, ..., `--node_rank 7` respectively.  `--master_addr` is set as the ip of the node 0.

- For example, to pre-train VideoMAE ViT-Base on **Kinetics400** with 64 GPUs (8 nodes x 8 GPUs), you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
  DATA_PATH='YOUR_PATH/list_kinetics-400/train.csv'
  
  OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 \
          --master_port 12320 --nnodes=8 \
          --node_rank=0 --master_addr=$your_node_0_ip \
          run_mae_pretraining.py \
          --data_path ${DATA_PATH} \
          --mask_type tube \
          --mask_ratio 0.9 \
          --model pretrain_videomae_base_patch16_224 \
          --decoder_depth 4 \
          --batch_size 32 \
          --num_frames 16 \
          --sampling_rate 4 \
          --opt adamw \
          --opt_betas 0.9 0.95 \
          --warmup_epochs 40 \
          --save_ckpt_freq 20 \
          --epochs 801 \
          --log_dir ${OUTPUT_DIR} \
          --output_dir ${OUTPUT_DIR}
  ```

  on the first node. On other nodes, run the same command with `--node_rank 1`, ..., `--node_rank 7` respectively.  `--master_addr` is set as the ip of the node 0.

### Note:

- Here the batch size is 32 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 2048.
- `lr` here is the base learning rate and is set to `1.5e-4` as default. The ` actual lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `` actual lr`` = `lr` * total batch size / 256.
- [Fixed]~~We have observed accidental interrupt in the last epoch when conduct the experiment on V100 GPUs (torch 1.6.0). This interrupt is caused by the scheduler of learning rate. We naively set  `--epochs 801` to walk away from issue :)~~

## Slurm

To help the community to reproduce our results on slurm cluster, we also provide the the **off-the-shelf** script. 

For example, to pre-train VideoMAE ViT-Base on **Kinetics400** with 64 GPUs (8 nodes x 8 GPUs), you can run

```bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
DATA_PATH='YOUR_PATH/list_kinetics-400/train.csv'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
```

