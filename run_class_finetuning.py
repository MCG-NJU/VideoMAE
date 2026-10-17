import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import  multiple_samples_collate
import utils
import modeling_finetune
from torch.utils.data import WeightedRandomSampler



#arguments
def get_args():
    
    parser = argparse.ArgumentParser('VideoMAE finetune - Single GPU, Windows')
    parser.add_argument('--finetune', default='', type=str,
                    help='path to pretrained VideoMAE checkpoint')

    # basic tarining
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--update_freq', default=1, type=int)

    # model
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int)

    parser.add_argument('--fc_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.0)  
    #finetune
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_segments', type=int, default=1)

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    # dataset
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--eval_data_path', default=None, type=str)


    parser.add_argument('--data_set', default='UCF-CRIME', type=str)

    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=1)
    parser.add_argument('--test_num_crop', type=int, default=1)

    parser.add_argument('--nb_classes', default=4, type=int)

    
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--device', default='cuda')

    return parser.parse_args(), None


#main 
def main(args, ds_init):

    print("Running in SINGLE GPU MODE (Windows). No Distributed Training.")
    print(args)

    device = torch.device("cuda")

    torch.manual_seed(42)
    np.random.seed(42)
    cudnn.benchmark = True

    # dataset
    print("Building datasets...")
    print(">>> DATA_PATH =", args.data_path)
    print(">>> DATA_SET =", args.data_set)

    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    #sampler
    labels = np.array(dataset_train.label_array)
    class_counts = np.bincount(labels, minlength=args.nb_classes)
    print("Train class distribution:", class_counts)

    inv_freq = 1.0 / (class_counts + 1e-6)
    inv_freq=inv_freq/inv_freq.sum()*len(inv_freq)

    sample_weights = inv_freq[labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler_train = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset_train) ,
        replacement=True
    )

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # dataloader
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    print("Train dataset size:", len(dataset_train))
    print("Train loader size:", len(data_loader_train))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    #model
    print("Building model:", args.model)

    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
    )

    model.to(device)

    # compute class weights from training dataset
    labels = dataset_train.label_array
    class_counts = np.bincount(labels, minlength=args.nb_classes)

    print("Class counts:", class_counts)

    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(class_counts)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("Class weights:", class_weights)

    if args.finetune and os.path.isfile(args.finetune):
        print(f">>> Loading pretrained weights from: {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location='cpu')
        checkpoint_model = checkpoint.get("model", checkpoint)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(">>> Load completed with message:", msg)
    else:
        print(">>> No pretrained checkpoint found, training from scratch.")

    # Freeze backbone AFTER loading checkpoint
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head")
    
    print("Backbone frozen, training only classification head.")
    total_blocks = model.get_num_layers()
    print(f"Model has {total_blocks} transformer blocks.")
    
    #optimizer and loss 
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    stage1_epochs = 10  
    total_epochs = args.epochs
    stage2_epochs = total_epochs - stage1_epochs
    loss_scaler=None
    #native scaler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,          # Head-only finetuning
        weight_decay=0.05
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=stage1_epochs,
    )

    #trainig loop
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs...")

    for epoch in range(args.epochs):

        #progressive un freez
        if epoch == stage1_epochs:
            print(">> Stage 2: Unfreezing last 2 blocks (blocks.10, blocks.11) + head")

            for name, param in model.named_parameters():
                if (
                    name.startswith("blocks.10") or
                    name.startswith("blocks.11") or
                    name.startswith("head")
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4,      
                weight_decay=0.05
            )

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=stage2_epochs,
            )

        if args.update_freq is None:
            args.update_freq = 1
        num_training_steps_per_epoch = len(data_loader_train)
        start_steps = epoch * num_training_steps_per_epoch

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=5.0,
            model_ema=None,
            mixup_fn=None,
            log_writer=None,
            start_steps=start_steps,
            lr_schedule_values=None,
            wd_schedule_values=None,
            num_training_steps_per_epoch=len(data_loader_train),
            update_freq=args.update_freq,   

        )
        if args.output_dir:
            if (epoch + 1) % 10 == 0:   # save every 10 epochs

                utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model,
                optimizer=optimizer,
                epoch=epoch
                )

        print(f'Epoch {epoch} Train Loss:', train_stats['loss'])
        lr_scheduler.step()
        print("Current LR:", optimizer.param_groups[0]["lr"])

        if not args.disable_eval_during_finetuning:
            val_stats = validation_one_epoch(data_loader_val, model, device)
            print(f"VAL ACC = {val_stats['acc1']:.2f}%")

    #final test 
    print("Running FINAL TEST...")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    preds_file = os.path.join(args.output_dir, "test_preds.txt")
    test_stats = final_test(data_loader_test, model, device, preds_file)

    print("TEST RESULTS SAVED TO:", preds_file)


#entry point 

if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)


    main(opts, ds_init)
