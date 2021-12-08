from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils
from losses import MaskLoss


def evaluate_performance(args, model, teacher_model, val_data_loader):

    running_loss, running_acc = 0.0, 0.0
    running_min_keep_ratio, running_avg_keep_ratio, running_max_keep_ratio = 1.0, 0.0, 0.0
    running_keep_ratios = []

    running_unpruned_acc = 0.0

    model.eval()
    teacher_model.eval()

    mask_loss_fn = MaskLoss(args, "val")

    metrics = {}
    accumulated_cls_attns = None

    # val_inputs, val_labels = next(iter(val_data_loader))
    # for _ in tqdm(range(1)):
    for val_inputs, val_labels in tqdm(val_data_loader):
        val_inputs = val_inputs.to(args.device)
        val_labels = val_labels.to(args.device)
        # val_inputs = mask_test_imgs
        # val_labels = mask_test_labels


        cls_attn_weights = teacher_model.forward_cls_attention(val_inputs.clone())  # (B, L, H, N+1)

        outputs = model(val_inputs.clone())
        model.unpruned = True
        unpruned_logits, _, _, _ = model(val_inputs.clone())
        model.unpruned = False
        unpruned_preds = torch.argmax(unpruned_logits.detach(), dim=1)
        running_unpruned_acc += torch.sum(unpruned_preds == val_labels).item() / val_labels.shape[0]

        logits, cls_attns, pred_logits, kept_token_idx = outputs

        mask_loss = mask_loss_fn(pred_logits, cls_attn_weights, kept_token_idx, metrics)

        loss = F.cross_entropy(logits, val_labels)
        preds = torch.argmax(logits.detach(), dim=1)

        # statistics
        running_loss += loss.detach().item()
        running_acc += torch.sum(preds == val_labels.data) / val_labels.shape[0]

        if args.patch_score_threshold is not None:
            running_keep_ratios += model.keep_ratios.tolist()
            running_min_keep_ratio = min(model.min_keep_ratio, running_min_keep_ratio)
            running_avg_keep_ratio += model.avg_keep_ratio
            running_max_keep_ratio = max(model.max_keep_ratio, running_max_keep_ratio)

    if args.patch_score_threshold is not None:
        attention_segmentation.dynamic_keep_ratio_hist(args, running_keep_ratios, "validation")
        metrics["val_min_keep_ratio"] = running_min_keep_ratio
        metrics["val_avg_keep_ratio"] = (running_avg_keep_ratio / len(val_data_loader))
        metrics["val_max_keep_ratio"] = running_max_keep_ratio

    metrics["val_loss"] = running_loss / len(val_data_loader)
    metrics["val_acc"] = float(running_acc) / len(val_data_loader)
    metrics["unpruned_acc"] = running_unpruned_acc / len(val_data_loader)

    args.epoch_acc = metrics['val_acc']  # for title of visualization plot
    print(f'val loss: {metrics["val_loss"]:.4f}, acc: {metrics["val_acc"]:.4f}')

    # for idx, cls_attns in enumerate(accumulated_cls_attns):
    #     # mean across batch
    #     cls_attns = torch.mean(cls_attns, dim=0)
    #     # average accumulated values over whole epoch
    #     cls_attns = cls_attns / len(val_data_loader)
    #     # round to 2 decimal places
    #     #cls_attns = torch.round(cls_attns * 10 ** 1) / (10 ** 1)
    #     accumulated_cls_attns[idx] = cls_attns

    # len(accumulated_cls_attns) = L
    # accumulated_cls_attns[0].shape = (H, N+1)

    return metrics #, accumulated_cls_attns


def evaluate_timing(args, model, teacher_model, val_data_loader):

    model.eval()
    teacher_model.eval()

    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)

    fwd_time, patch_emb_time, attn_time, drop1_time, mlp_time, drop2_time, encoder_time, head_time, pred_time, \
        pure_attn_time = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    teacher_fwd_time, teacher_patch_emb_time, teacher_encoder_time, teacher_attn_time, teacher_drop1_time, \
        teacher_mlp_time, teacher_drop2_time, teacher_head_time, teacher_pure_attn_time = 0, 0, 0, 0, 0, 0, 0, 0, 0

    for val_idxs, val_attn_weights, val_inputs, val_labels in tqdm(val_data_loader):
        val_attn_weights = val_attn_weights.to(args.device)
        val_inputs = val_inputs.to(args.device)

        fwd_start.record()
        _ = teacher_model(val_inputs.clone())
        fwd_end.record()
        torch.cuda.synchronize()
        teacher_fwd_time += fwd_start.elapsed_time(fwd_end)
        teacher_patch_emb_time += teacher_model.patch_emb_start.elapsed_time(teacher_model.patch_emb_end)
        teacher_encoder_time += teacher_model.encoder_start.elapsed_time(teacher_model.encoder_end)
        teacher_head_time += teacher_model.head_start.elapsed_time(teacher_model.head_end)
        for blk in list(teacher_model.children())[2]:
            teacher_attn_time += blk.attn_start.elapsed_time(blk.attn_end)
            teacher_drop1_time += blk.drop1_start.elapsed_time(blk.drop1_end)
            teacher_mlp_time += blk.mlp_start.elapsed_time(blk.mlp_end)
            teacher_drop2_time += blk.drop2_start.elapsed_time(blk.drop2_end)
            teacher_pure_attn_time += blk.attn.attn_start.elapsed_time(blk.attn.attn_end)

        fwd_start.record()
        _ = model(val_inputs.clone())
        fwd_end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        fwd_time += fwd_start.elapsed_time(fwd_end)
        patch_emb_time += model.patch_emb_start.elapsed_time(model.patch_emb_end)
        encoder_time += model.encoder_start.elapsed_time(model.encoder_end)
        head_time += model.head_start.elapsed_time(model.head_end)
        for blk in list(model.children())[2]:
            attn_time += blk.attn_start.elapsed_time(blk.attn_end)
            drop1_time += blk.drop1_start.elapsed_time(blk.drop1_end)
            mlp_time += blk.mlp_start.elapsed_time(blk.mlp_end)
            drop2_time += blk.drop2_start.elapsed_time(blk.drop2_end)
            pure_attn_time += blk.attn.attn_start.elapsed_time(blk.attn.attn_end)
        # get saved forward time from model's predictor submodule fwd_time attribute
        pred_time += model.pred_start.elapsed_time(model.pred_end)

    teacher_fwd_time /= len(val_data_loader)
    teacher_patch_emb_time /= len(val_data_loader)
    teacher_encoder_time /= len(val_data_loader)
    teacher_head_time /= len(val_data_loader)
    teacher_attn_time /= len(val_data_loader)
    teacher_drop1_time /= len(val_data_loader)
    teacher_mlp_time /= len(val_data_loader)
    teacher_drop2_time /= len(val_data_loader)
    teacher_pure_attn_time /= len(val_data_loader)

    fwd_time /= len(val_data_loader)
    pred_time /= len(val_data_loader)
    patch_emb_time /= len(val_data_loader)
    encoder_time /= len(val_data_loader)
    head_time /= len(val_data_loader)
    attn_time /= len(val_data_loader)
    drop1_time /= len(val_data_loader)
    mlp_time /= len(val_data_loader)
    drop2_time /= len(val_data_loader)
    pure_attn_time /= len(val_data_loader)

    print(f'avg unpruned forward pass took {teacher_fwd_time:.2f} ms, '
          f'avg unpruned patch embedding took {teacher_patch_emb_time:.2f} ms, '
          f'avg unpruned encoder took {teacher_encoder_time:.2f} ms, '
          f'avg unpruned MHSA block took {teacher_attn_time:.2f} ms, '
          f'avg unpruned pure attention took {teacher_pure_attn_time:.2f} ms, '
          f'avg unpruned dropout 1 took {teacher_drop1_time:.2f} ms, '
          f'avg unpruned MLP block took {teacher_mlp_time:.2f} ms, '
          f'avg unpruned dropout 2 took {teacher_drop2_time:.2f} ms, '
          f'avg unpruned classifier head took {teacher_head_time:.2f} ms\n')

    print(f'avg forward pass took {fwd_time:.2f} ms, '
          f'avg patch embedding took {patch_emb_time:.2f} ms, '
          f'avg encoder took {encoder_time:.2f} ms, '
          f'avg predictor took {pred_time:.2f} ms, '
          f'avg MHSA block took {attn_time:.2f} ms, '
          f'avg pure attention took {pure_attn_time:.2f} ms, '
          f'avg dropout 1 took {drop1_time:.2f} ms, '
          f'avg MLP block took {mlp_time:.2f} ms, '
          f'avg dropout 2 took {drop2_time:.2f} ms, '
         f'avg classifier head took {head_time:.2f} ms')

    return