import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import torch.amp

def train_class_batch(model, samples, target, criterion):       
    
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, 
                    mixup_fn: Optional[Mixup] = None, 
                    log_writer=None,
                    start_steps=None, 
                    lr_schedule_values=None, 
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None, 
                    update_freq=1,
                    ):

    model.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue

        it = start_steps + step

        # LR & WD scheduling
        if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

#forward
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(loss_scaler is not None)):
            loss, output = train_class_batch(model, samples, targets, criterion)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            sys.exit(1)

        # gradient accumulation
        loss = loss / update_freq

#backward
        if loss_scaler is not None:
            # AMP backward + optimizer.step() automatically
            loss_scale_value = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=False
            )
            grad_norm = None
        else:
            # FP32 fallback
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            grad_norm = None
            loss_scale_value = 1.0

        # EMA
        if model_ema is not None and (data_iter_step + 1) % update_freq == 0:
            model_ema.update(model)

        torch.cuda.synchronize()

        # Accuracy logging
        if mixup_fn is None:
            class_acc = (output.argmax(dim=-1) == targets).float().mean()
        else:
            class_acc = None

        # Logging
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = min(pg["lr"] for pg in optimizer.param_groups)
        max_lr = max(pg["lr"] for pg in optimizer.param_groups)
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        wd = next((pg["weight_decay"] for pg in optimizer.param_groups if pg["weight_decay"] > 0), None)
        metric_logger.update(weight_decay=wd)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=wd, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)

        # FP32 — remove AMP
        output = model(videos)
        #testing 
        pred = output.argmax(dim=1)

        print("PRED:", pred.cpu().numpy())
        print("GT:", target.cpu().numpy())
        print("LOGITS:", output[0].detach().cpu().numpy())
        print("------")
        
        loss = criterion(output, target)
        num_classes = output.size(-1)
        topk = (1, min(5, num_classes))
        acc1, acc5 = accuracy(output, target, topk=topk)

        batch_size = videos.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} '
          f'Acc@5 {metric_logger.acc5.global_avg:.3f} '
          f'loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]

        # FP32 forward
        output = model(videos)
        loss = criterion(output, target)

        for i in range(output.size(0)):
            s = f"{ids[i]} {output[i].cpu().numpy().tolist()} {int(target[i])} {int(chunk_nb[i])} {int(split_nb[i])}\n"
            final_result.append(s)

        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1,))[0]
        acc5 = torch.tensor(0.0)  

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # save results
    if not os.path.exists(file):
        open(file, 'w').close()

    with open(file, 'w') as f:
        f.write(f"{acc1}, {acc5}\n")
        for line in final_result:
            f.write(line)

    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} '
          f'Acc@5 {metric_logger.acc5.global_avg:.3f} '
          f'loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}

    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, f"{x}.txt")
        lines = open(file, 'r').readlines()[1:]

        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=float, sep=',')
            data = softmax(data)

            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []

            if chunk_nb + split_nb in dict_pos[name]:
                continue

            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label

    print("Computing final results")

    input_lst = []
    print(len(dict_feats))

    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])

    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)

    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]

    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = np.mean([x for x in data], axis=0)
    pred = np.argmax(feat)
    top1 = 1.0 if int(pred) == int(label) else 0.0
    top5 = 1.0 if int(label) in np.argsort(-feat)[:5] else 0.0
    return [pred, top1, top5, int(label)]
