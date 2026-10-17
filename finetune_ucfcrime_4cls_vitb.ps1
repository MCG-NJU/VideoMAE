$env:DATA_PATH='C:\Users\Nekou\Desktop\final_project\final_project\VideoMAE\codes\VideoMAE\list_ucfcrime_4cls'
$env:FINETUNE_CKPT='C:\Users\Nekou\Desktop\final_project\final_project\VideoMAE\codes\VideoMAE\pretrained\k400_vitb_e800_pretrain.pth'
$env:OUTPUT_DIR='C:\Users\Nekou\Desktop\final_project\final_project\VideoMAE\codes\VideoMAE\outputs\ucfcrime_4cls_vitb'

mkdir $env:OUTPUT_DIR -Force | Out_Null

python .\run_class_finetuning.py `
    --model vit_base_patch16_224 `
    --data_set UCF101 `
    --nb_classes 4 `
    --data_path $env:DATA_PATH `
    --finetune $env:FINETUNE_CKPT `
    --log_dir $env:OUTPUT_DIR `
    --device cuda `
    --batch_size 4 `
    --num_frames 16 `
    --sampling_rate 4 `
    --num_sample 1 `
    --short_side_size 224 `
    --epochs 30 `
    --opt adamw `
    --lr 1e-4 `
    --warmup_epochs 5 `
    --warmup_steps -1 `
    --mixup 0 `
    --cutmix 0 `
    --test_num_crop 1 `
    --num_workers 2