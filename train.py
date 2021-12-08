import torch
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm
import utils
from losses import MaskLoss, BackboneLoss


def train_one_epoch(args, model, teacher_model, train_data_loader, optimizer, mixup_fn=None):

    running_loss, running_min_keep_ratio, running_avg_keep_ratio, running_max_keep_ratio = 0.0, 1.0, 0.0, 0.0
    running_keep_ratios = []

    mask_loss_fn = MaskLoss(args, "train")
    backbone_loss_fn = BackboneLoss(args)

    metrics = {}

    model.train()
    teacher_model.eval()

    # train_inputs, train_labels = next(iter(train_data_loader))
    # train_inputs = train_inputs.to(args.device)
    # train_labels = train_labels.to(args.device)
    # for train_step in tqdm(range(1)):
    for train_step, train_data in enumerate(tqdm(train_data_loader)):
        train_inputs = train_data[0].to(args.device)
        train_labels = train_data[1].to(args.device)

        if mixup_fn is not None:
            train_inputs, train_labels = mixup_fn(train_inputs, train_labels)

        # flops_unpruned = FlopCountAnalysis(teacher_model, train_inputs.clone())
        # flops_pruned = FlopCountAnalysis(model, train_inputs.clone())
        # print(f"Flops [GFLOPs]: Unpruned={flops_unpruned.total() / 1e9:.2f}, Pruned={flops_pruned.total() / 1e9:.2f}, "
        #      f"Ratio={flops_pruned.total() / flops_unpruned.total():.2f}")
        # exit()

        # logits from classifier head, final token representation, and CLS attention weights from teacher
        logits_t, token_t, cls_attn_weights = teacher_model(train_inputs.clone())  # (B, L, H, N+1)

        # forward
        logits_s, token_s, pred_logits, kept_token_idx = model(train_inputs.clone()) #token_rec

        # predictor network in form of a MLP
        mask_loss = mask_loss_fn(pred_logits, cls_attn_weights, kept_token_idx, metrics)

        backbone_loss = backbone_loss_fn(logits_s, token_s, logits_t, token_t, kept_token_idx, train_labels, metrics)

        if args.step < args.warmup_steps:  # or args.step % 2 == 1:
            train_loss = mask_loss  # + reconstruction_loss
        else:
            train_loss = backbone_loss + mask_loss  # + reconstruction_loss
        # zero the parameter gradients
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if train_step % (400 if args.is_sbatch else 10) == 0:
            print(f'training step_{train_step} '
                  f'mask loss: {mask_loss.detach().item():.4f}, '
                  f'train loss: {train_loss.detach().item():.4f}, ')

        running_loss += train_loss.detach().item()

        if args.patch_score_threshold is not None:
            running_keep_ratios += model.keep_ratios.tolist()
            running_min_keep_ratio = min(model.min_keep_ratio, running_min_keep_ratio)
            running_avg_keep_ratio += model.avg_keep_ratio
            running_max_keep_ratio = max(model.max_keep_ratio, running_max_keep_ratio)
    

    # keep_ratios_table = wandb.Table(data=running_keep_ratios, columns=["keep_ratios"])
    # metrics["train_keep_ratios_hist"] = wandb.plot.histogram(keep_ratios_table, "keep_ratios", title="Training Keep Ratios"
    #                                                                                                  "Histogram")
    if args.patch_score_threshold is not None:
        attention_segmentation.dynamic_keep_ratio_hist(args, running_keep_ratios, "training")
        metrics["train_min_keep_ratio"] = running_min_keep_ratio
        metrics["train_avg_keep_ratio"] = (running_avg_keep_ratio / len(train_data_loader))
        metrics["train_max_keep_ratio"] = running_max_keep_ratio

    metrics["train_loss"] = running_loss / len(train_data_loader)
    print(f'train loss: {metrics["train_loss"]:.4f}')

    return metrics
