import torch
from torch.nn import functional as F
from timm.loss import SoftTargetCrossEntropy


class MaskLoss(torch.nn.Module):
    def __init__(self, args, phase):
        super().__init__()
        self.phase = phase
        self.keep_ratios = args.keep_ratios
        self.loss_type = args.mask_loss_type
        self.count = 1
        self.running_loss = 0
        self.runnings_accs = [0 for _ in self.keep_ratios]

        if self.loss_type == "bce":
            # for pixel-wise crossentropy loss
            # as the classes (1: kept token, 0: dropped token) we use a weight for the positive class to counteract
            # this class imbalance
            mask_loss_fns = []
            for i, ratio in enumerate(args.keep_ratios):
                if i > 0:
                    curr_ratio = ratio / args.keep_ratios[i - 1]
                else:
                    curr_ratio = args.keep_ratios[i]
                dropped_token_weights = torch.ones(size=(args.batch_size,)) * curr_ratio / (1 - curr_ratio)
                kept_token_weights = torch.ones(size=(args.batch_size,)) * (1 - curr_ratio) / curr_ratio
                weights = torch.cat((dropped_token_weights, kept_token_weights)).to(args.device)
                mask_loss_fns.append(torch.nn.BCEWithLogitsLoss(weight=weights[1], reduction='mean'))
            self.bce_loss_fns = mask_loss_fns

    def forward(self, pred_logits, cls_attn_weights, kept_token_idx, metrics):

        mask_loss = 0
        mask_accs = [0 for _ in self.keep_ratios]
        if self.loss_type == "bce":
            # 0 in predicted logits --> negative class/dropped patches, 1 in predicted logits --> positive class/kept patches
            # 0 in predicted mask/ground truth mask --> drop patch, 1 in predicted mask/ground truth mask --> keep patch
            # 0 in patch wise labels --> negative class/dropped patches, 1 in patch wise labels --> positive class/kept patches
            # gt_patch_drop_mask:  0 --> dropped token, 1 --> kept token
            # labels: 0 --> class 0 or dropped token, 1 --> class 1 or kept token
            # BCE div loss for keep mask prediction task
            cls_attn_weights = torch.mean(cls_attn_weights, dim=1)  # (B, H, N+1)
            cls_attn_weights, _ = torch.max(cls_attn_weights, dim=1)  # (B, N+1)
            renormalized_cls = cls_attn_weights[:, 1:] / torch.sum(cls_attn_weights[:, 1:], dim=-1,
                                                                   keepdim=True)
            for i in range(len(kept_token_idx)):
                if i > 0:
                    gt_patch_drop_mask = self.get_mask_from_cls_attns(torch.gather(renormalized_cls, 1,
                                                                              kept_token_idx[i - 1]),
                                                                 self.keep_ratios[i] / self.keep_ratios[i - 1])
                    pred_keep_mask = self.get_mask_from_pred_logits(pred_logits[i],
                                                               self.keep_ratios[i] / self.keep_ratios[i - 1])
                else:
                    gt_patch_drop_mask = self.get_mask_from_cls_attns(renormalized_cls, self.keep_ratios[i])
                    pred_keep_mask = self.get_mask_from_pred_logits(pred_logits[i], self.keep_ratios[i])
                patch_wise_labels = gt_patch_drop_mask.long().to(args.device)
                mask_loss += self.mask_criterions[i](pred_logits[i].flatten(start_dim=0, end_dim=1),
                                                patch_wise_labels.float().flatten())
                mask_accs[i] += torch.sum(pred_keep_mask == gt_patch_drop_mask) / pred_keep_mask.numel()
        elif self.loss_type == "mse":
            # MSE loss for keep mask prediction task
            cls_attn_weights = torch.mean(cls_attn_weights, dim=1)  # (B, H, N+1)
            cls_attn_weights, _ = torch.max(cls_attn_weights, dim=1)  # (B, N+1)
            renormalized_cls = cls_attn_weights[:, 1:] / torch.sum(cls_attn_weights[:, 1:], dim=-1,
                                                                   keepdim=True)
            pred_keep_mask = self.get_mask_from_pred_logits(F.softmax(pred_logits[0], dim=-1), self.keep_ratios[0])
            for i in range(len(kept_token_idx)):
                # KL div loss for keep mask prediction task
                temp = 1e-2
                if i > 0:
                    renormalized_cls = torch.gather(input=renormalized_cls, dim=1, index=kept_token_idx[i - 1])
                    renormalized_cls /= torch.sum(renormalized_cls, dim=1, keepdim=True)
                mask_loss += 100 * F.mse_loss(pred_logits[i], renormalized_cls, reduction='mean')
        else:
            cls_attn_weights = torch.mean(cls_attn_weights, dim=1)  # (B, H, N+1)
            cls_attn_weights, _ = torch.max(cls_attn_weights, dim=1)  # (B, N+1)
            renormalized_cls = cls_attn_weights[:, 1:] / torch.sum(cls_attn_weights[:, 1:], dim=-1,
                                                                   keepdim=True)
            for i in range(len(kept_token_idx)):
                # KL div loss for keep mask prediction task
                temp = 1e-2
                if i > 0:
                    gt_patch_drop_mask = self.get_mask_from_cls_attns(torch.gather(renormalized_cls, 1,
                                                                              kept_token_idx[i - 1]),
                                                                 self.keep_ratios[i] / self.keep_ratios[i - 1])
                    pred_keep_mask = self.get_mask_from_pred_logits(F.softmax(pred_logits[i], dim=-1),
                                                               self.keep_ratios[i] / self.keep_ratios[i - 1])
                    renormalized_cls = torch.gather(input=renormalized_cls, dim=1, index=kept_token_idx[i - 1])
                    renormalized_cls /= torch.sum(renormalized_cls, dim=1, keepdim=True)
                else:
                    gt_patch_drop_mask = self.get_mask_from_cls_attns(renormalized_cls, self.keep_ratios[i])
                    pred_keep_mask = self.get_mask_from_pred_logits(F.softmax(pred_logits[i], dim=-1), self.keep_ratios[i])
                mask_loss += F.kl_div(F.log_softmax(pred_logits[i], dim=-1), torch.log(renormalized_cls),
                                      log_target=True, reduction='batchmean')
                mask_accs[i] += torch.sum(pred_keep_mask == gt_patch_drop_mask) / pred_keep_mask.numel()

        # TP += torch.sum(patch_wise_labels[pred_keep_mask == 1] == 1).item()  # keep patch predicted for keep patch class
        # TN += torch.sum(patch_wise_labels[pred_keep_mask == 0] == 0).item()  # drop patch predicted for drop patch class
        # FP += torch.sum(patch_wise_labels[pred_keep_mask == 1] == 0).item()  # keep patch predicted for drop patch class
        # FN += torch.sum(patch_wise_labels[pred_keep_mask == 0] == 1).item()  # drop patch predicted for keep patch class

        # metrics[f"{phase} TP"] = TP
        # metrics[f"{phase} TN"] = TN
        # metrics[f"{phase} FP"] = FP
        # metrics[f"{phase} FN"] = FN
        # metrics[f"{phase} FPR"] = FP / (FP + TN)  # False Positive Rate
        # metrics[f"{phase} Recall"] = TP / (TP + FN)  # True Positive Rate
        # metrics[f"{phase} Precision"] = TP / (TP + FP)  # Positive Predictive Value

        self.running_loss += mask_loss.detach().item()
        metrics[f"{self.phase}_mask_loss"] = self.running_loss / self.count
        for i, _ in enumerate(self.keep_ratios):
            self.runnings_accs[i] += mask_accs[i]
            metrics[f"{self.phase}_mask_acc_{i}"] = self.runnings_accs[i] / self.count

        self.count += 1

        return mask_loss

    @staticmethod
    def get_mask_from_pred_logits(logits, keep_ratio):
        """
            input: logits, (B, N) the predicted scores for each token in the token sequences in the current batch
            keep_ratio: the amount of tokens to keep in percent, e.g. [0,1]
            mean_heads: whether to aggregate the attention weights from the different heads by averaging or taking the max
                        across the attention heads
        """

        sort_idxs = torch.argsort(logits, dim=-1, descending=True)

        num_kept_tokens = int(logits.shape[-1] * keep_ratio)
        kept_mask = torch.ones_like(sort_idxs[:, :num_kept_tokens], device=logits.device)
        dropped_mask = torch.zeros_like(sort_idxs[:, num_kept_tokens:], device=logits.device)
        mask = torch.cat((kept_mask, dropped_mask), dim=-1).float()

        mask.scatter_(index=sort_idxs, src=mask.clone(), dim=-1)

        return mask

    @staticmethod
    def get_mask_from_cls_attns(cls_attns, keep_ratio):
        """
            input: cls_attns, (B, N) the CLS attention weights from the unpruned teacher averaged over all layers and
                                        aggregated over all heads
            keep_ratio: the amount of tokens to keep in percent, e.g. [0,1]
            mean_heads: whether to aggregate the attention weights from the different heads by averaging or taking the max
                        across the attention heads
        """
        # sort in order to take K highest according to keeping ratio
        sort_idxs = torch.argsort(cls_attns, dim=-1, descending=True)

        # compute nubmer of kept tokens
        num_kept_tokens = int(cls_attns.shape[-1] * keep_ratio)
        # 1s in mask --> kept tokens
        kept_mask = torch.ones_like(sort_idxs[:, :num_kept_tokens], device=cls_attns.device)
        # 0s in mask --> dropped tokens
        dropped_mask = torch.zeros_like(sort_idxs[:, num_kept_tokens:], device=cls_attns.device)
        mask = torch.cat((kept_mask, dropped_mask), dim=-1).float()

        # bring back tokens in original order (currently still sorted descending)
        mask.scatter_(index=sort_idxs, src=mask.clone(), dim=-1)

        return mask


class BackboneLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.mixup > 0.:
            # smoothing is handled with mixup label transform
            base_criterion = SoftTargetCrossEntropy()
        else:
            base_criterion = torch.nn.CrossEntropyLoss()
        self.base_criterion = base_criterion

        self.patch_score_threshold = args.patch_score_threshold
        self.count = 1
        self.running_loss = 0
        self.running_cls_loss = 0
        self.running_token_kl_loss = 0
        self.running_token_dist_loss = 0
        self.runnings_acc = 0

    def forward(self, logits_s, token_s, logits_t, token_t, kept_token_idx, train_labels, metrics):

        # reconstruction_loss = 0
        # for i, idx in enumerate(model.dropped_token_indices):
        #     gt_token = torch.gather(token_t, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, token_t.shape[-1]))
        #     assert gt_token.shape == token_rec[i].shape
        #     gt_token = gt_token.flatten()
        #     token_rec[i] = token_rec[i].flatten()
        #     reconstruction_loss += F.mse_loss(gt_token, token_rec[i], reduction='mean')
        # reconstruction_loss /= len(model.dropped_token_indices)

        cls_loss = self.base_criterion(logits_s, train_labels)

        cls_kl_loss = F.kl_div(
            F.log_softmax(logits_s, dim=-1),
            F.log_softmax(logits_t, dim=-1),
            reduction='batchmean',
            log_target=True
        )
        if self.patch_score_threshold is None:
            # token_t = torch.gather(input=token_t, dim=1,
            #                        index=kept_token_idx.unsqueeze(-1).expand(-1, -1, token_t.shape[-1]))
            B, N, C = token_t.size()
            # kept_token_idx = kept_token_idx.reshape(B * N).long()
            # token_s = token_s.reshape(-1, C)
            # token_t = token_t.reshape(B * N, C)
            # token_s = token_s[kept_token_idx]
            token_t = torch.gather(input=token_t, dim=1, index=kept_token_idx[-1].unsqueeze(-1).expand(-1, -1, C))
            # discard the aggregation tokens which are the sum of the dropped tokens at each pruning stage
            # token_s = token_s[:, :-(1+len(args.keep_ratios)-1)]
        else:
            token_t = token_t.flatten()[kept_token_idx[-1]]

        token_s = token_s.reshape(-1, C)
        token_t = token_t.reshape(-1, C)
        token_kl_loss = F.kl_div(
            F.log_softmax(token_s, dim=-1),
            F.log_softmax(token_t, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        backbone_loss = cls_loss + cls_kl_loss + token_kl_loss

        # statistics
        self.running_loss += backbone_loss.detach().item()
        self.running_cls_loss += cls_loss.detach().item()
        self.running_token_dist_loss += cls_kl_loss.detach().item()
        self.running_token_kl_loss += token_kl_loss.detach().item()

        # metrics["train_reconstruction_loss"] = running_recon_loss / len(train_data_loader)
        metrics["train_backbone_loss"] = self.running_loss / self.count
        metrics["train_cls_loss"] = self.running_cls_loss / self.count
        metrics["train_token_kl_loss"] = self.running_token_dist_loss / self.count
        metrics["train_cls_kl_loss"] = self.running_token_kl_loss / self.count
        self.count += 1

        return backbone_loss
