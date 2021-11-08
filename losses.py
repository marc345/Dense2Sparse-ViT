import torch
from torch.nn import functional as F

class DistillDiffPruningLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, args, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5,
                 dynamic=False, pruning_loc=[3], keep_ratio=[0.3], clf_weight=1.0, mse_token=False, softmax_temp=1.0,
                 print_mode=True, early_exit=False):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.early_exit_cls_loss = 0
        self.early_exit_cls_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

        self.topk_selection = args.topk_selection
        self.use_ratio_loss = args.use_ratio_loss
        self.use_token_dist_loss = args.use_token_dist_loss
        self.use_teacher_cls_loss = args.teacher_cls_loss

        if self.use_teacher_cls_loss:
            # for pixel-wise crossentropy loss
            # as the classes (1: kept token, 0: dropped token) we use a weight for the positive class to counteract
            # this class imbalance
            kept_token_weights = torch.ones(size=(1,)) * (1 - args.keep_ratios[0]) / args.keep_ratios[0]
            dropped_token_weights = torch.ones_like(kept_token_weights)
            weights = torch.cat((kept_token_weights, dropped_token_weights)).to(args.device)
            self.bce_loss_teacher_cls = torch.nn.CrossEntropyLoss(weight=weights)

        self.mean_heads = args.mean_heads

        # temperature value used for distillation loss parts
        self.T = softmax_temp

        self.early_exit = early_exit

        print('cls_weight=', clf_weight, 'distill_weight=', distill_weight, 'softmax_temp=', softmax_temp)
        if not self.topk_selection and self.use_token_dist_loss:
            print('using KL divergence of final layer tokens in loss function (weight: see distill_weight above)')
        if not self.topk_selection and self.use_ratio_loss:
            print(f'using ratio of kept tokens in loss function, weight={ratio_weight}')

        if dynamic:
            print('using dynamic loss')

        if self.topk_selection:
            print('using differentiable top-k instead of gumbel softmax for patch importance selection')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        if self.topk_selection:
            # class probabilities, final layer tokens (without CLS), indices of kept input data
            student_logits, predictor_logits, spatial_tokens = outputs
        elif self.early_exit:
            # predictions, final layer tokens (without CLS), final token keep decisions, all token keep decisions
            student_logits, early_exit_pred, spatial_tokens, mask, predictor_keep_prob = outputs
        else:
            student_logits, spatial_tokens, mask, predictor_logits, predictor_keep_prob = outputs

        # crossentropy loss between student predictions and ground truth labels
        cls_loss = self.base_criterion(student_logits, labels)

        with torch.no_grad():
            if self.use_teacher_cls_loss:
                teacher_logits, token_t, cls_attn_weights = self.teacher_model(
                    inputs, return_cls_attns=True)  # 3rd return value is final CLS token attention weights
                gt_patch_drop_mask = get_mask_from_cls_attns(cls_attn_weights, args.keep_ratios[0],
                                                             mean_heads=self.mean_heads)

                num_same_mask_elements = torch.sum(pred_keep_mask.detach() == gt_patch_drop_mask.detach())
                num_different_mask_elements = torch.sum(pred_keep_mask.detach() != gt_patch_drop_mask.detach())
                total_mask_elements = num_same_mask_elements + num_different_mask_elements
                assert (total_mask_elements == pred_keep_mask.numel()), "Number of same and different elements in both " \
                                                                        "masks do not sum up to the total number of patches"

                # ground truth drop mask: 0 --> drop, 1--> keep
                # labels/classes for bce loss function: 0--> keep, 1--> drop
                patch_wise_labels = (1 - gt_patch_drop_mask).long().to(args.device)
                mask_ce_loss = self.bce_loss_teacher_cls(pred_scores.flatten(start_dim=0, end_dim=1),
                                                         patch_wise_labels.flatten())
                if i % 50 == 0:
                    # print(f'training step_{i} mask_mse_loss: {mse_loss.detach().item():.4f}, '
                    #       f'cls_loss: {cls_loss.detach().item():.4f}, '
                    #       f'acc: {(torch.sum(preds == train_labels.data) / train_labels.shape[0]).item():.4f}')
                    print(f'training step_{i} pixel_wise_ce_loss: {train_loss.detach().item():.4f}, '
                          f'cls_loss: {cls_loss.detach().item():.4f}, '
                          f'acc: {(torch.sum(preds == train_labels.data) / train_labels.shape[0]).item():.4f}')
                    print(f'Number of elements where masks differ for whole batch: {num_different_mask_elements}')
            else:
                # teacher logits and spatial tokens
                teacher_logits, token_t = self.teacher_model(inputs)  # 3rd return value is final CLS token attention weights

        # KL divergence between student class probabilities and teacher class probabilities
        cls_kl_loss = F.kl_div(
            F.log_softmax(student_logits/self.T, dim=-1),
            F.log_softmax(teacher_logits/self.T, dim=-1),
            reduction='batchmean',
            log_target=True
        ) * (self.T ** 2)

        # early exit losses computed from CLS token predictions of student network in the layer before the pruning stage
        if self.early_exit:
            # cross entropy loss between student early exit predictions from cls token and ground truth labels
            early_exit_cls_loss = self.base_criterion(early_exit_pred, labels)

            # KL divergence between early exit student class probabilities and teacher class probabilities
            early_exit_cls_distill_loss = F.kl_div(
                F.log_softmax(early_exit_pred/self.T, dim=-1),
                F.log_softmax(teacher_logits/self.T, dim=-1),
                reduction='batchmean',
                log_target=True
            ) * (self.T ** 2)

        if self.topk_selection:
            #token_kl_loss = F.kl_div(
            #    F.log_softmax(spatial_tokens, dim=-1),
            #    F.log_softmax(token_t, dim=-1),
            #    reduction='batchmean',
            #    log_target=True
            #)
            loss = self.clf_weight * cls_loss + self.distill_weight * cls_kl_loss #+ self.distill_weight * token_kl_loss

        else:
            if self.use_ratio_loss:
                # ratio loss
                pred_loss = 0.0
                ratio = self.keep_ratio
                for i, score in enumerate(predictor_keep_prob):
                    if self.dynamic:
                        pos_ratio = score.mean()
                    else:
                        pos_ratio = score.mean(1)
                    pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

            if self.use_token_dist_loss:
                # final token distillation loss
                B, N, C = spatial_tokens.size()
                assert mask.numel() == B * N
                bool_mask = mask.reshape(B * N) > 0.5
                spatial_tokens = spatial_tokens.reshape(B * N, C)
                token_t = token_t.reshape(B * N, C)
                if mask.sum() < 0.1:
                    token_kl_loss = spatial_tokens.new(1, ).fill_(0.0)
                else:
                    token_t = token_t[bool_mask]
                    spatial_tokens = spatial_tokens[bool_mask]
                    if self.mse_token:
                        token_kl_loss = torch.pow(spatial_tokens - token_t, 2).mean()
                    else:
                        # KL divergence between final encoder layer tokens (CLS token excluded)
                        token_kl_loss = F.kl_div(
                            F.log_softmax(spatial_tokens, dim=-1),
                            F.log_softmax(token_t, dim=-1),
                            reduction='batchmean',
                            log_target=True
                        )

            # print(cls_loss, pred_loss)
            loss = self.clf_weight * cls_loss + self.distill_weight * cls_kl_loss
            if self.early_exit:
                loss += early_exit_cls_loss + early_exit_cls_distill_loss
            if self.use_ratio_loss:
                loss += self.ratio_weight * pred_loss / len(self.pruning_loc)
            if self.use_token_dist_loss:
                loss += self.distill_weight * token_kl_loss

        if self.print_mode:
            self.cls_loss += cls_loss.detach().item() * self.clf_weight
            self.cls_distill_loss += cls_kl_loss.detach().item() * self.distill_weight

            if self.early_exit:
                self.early_exit_cls_loss += early_exit_cls_loss.detach().item()
                self.early_exit_cls_distill_loss += early_exit_cls_distill_loss.detach().item()

            if not self.topk_selection and self.use_ratio_loss:
                self.ratio_loss += pred_loss.detach().item() * self.ratio_weight
            if not self.topk_selection and self.use_token_dist_loss:
                self.token_distill_loss += token_kl_loss.detach().item() * self.distill_weight

            self.count += 1
            if self.count % 100 == 1:
                if self.topk_selection:
                    print(f'loss info: cls_loss={(self.cls_loss / self.count):.4f}, '
                          f'cls_kl={(self.cls_distill_loss / self.count):.4f}')
                else:
                    loss_info = f'loss info: cls_loss={(self.cls_loss / self.count):.4f},   ' \
                                f'cls_kl={(self.cls_distill_loss / self.count):.4f}'
                    if self.early_exit:
                        loss_info += f'   ee_cls_loss={(self.early_exit_cls_loss / self.count):.4f},    ' \
                                     f'ee_cls_kl={(self.early_exit_cls_distill_loss / self.count):.4f}'
                    if self.use_ratio_loss:
                        loss_info += f',   ratio loss={(self.ratio_loss/self.count):.4f}'
                    if self.use_token_dist_loss:
                        loss_info += f',   final_token_kl={(self.token_distill_loss/self.count):.4f}'
                    print(loss_info)
        return loss

    @staticmethod
    def get_mask_from_pred_scores(pred_scores, keep_ratio):
        """
            input: pred_scores, (B, N) the predicted scores for each token in the token sequences in the current batch
            keep_ratio: the amount of tokens to keep in percent, e.g. [0,1]
            mean_heads: whether to aggregate the attention weights from the different heads by averaging or taking the max
                        across the attention heads
        """

        sort_idxs = torch.argsort(pred_scores, dim=-1, descending=True)

        num_kept_tokens = int(pred_scores.shape[-1] * keep_ratio)
        kept_mask = torch.ones_like(sort_idxs[:, :num_kept_tokens], device=pred_scores.device)
        dropped_mask = torch.zeros_like(sort_idxs[:, num_kept_tokens:], device=pred_scores.device)
        mask = torch.cat((kept_mask, dropped_mask), dim=-1).float()

        mask.scatter_(index=sort_idxs, src=mask.clone(), dim=-1)

        return mask


    @staticmethod
    def get_mask_from_cls_attns(cls_attns, keep_ratio, mean_heads=False):
        """
            input: cls_attns, (B, L, H, N+1) the CLS attention weights from the unpruned teacher network from all the
                   encoder layers and different attention heads
            keep_ratio: the amount of tokens to keep in percent, e.g. [0,1]
            mean_heads: whether to aggregate the attention weights from the different heads by averaging or taking the max
                        across the attention heads
        """
        # mean across all encoder layers
        cls_attns = torch.mean(cls_attns, dim=1)
        if mean_heads:
            # aggregate across heads via mean
            cls_attns = torch.mean(cls_attns, dim=1)
        else:
            # aggregate across heads via max
            cls_attns, _ = torch.max(cls_attns, dim=1)
        # exclude CLS weight
        cls_attns = cls_attns[:, 1:]

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