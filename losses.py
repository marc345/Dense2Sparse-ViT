import torch
from torch.nn import functional as F

class DistillDiffPruningLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, args, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5,
                 dynamic=False, pruning_loc=[3], keep_ratio=[0.3], clf_weight=1.0, mse_token=False,
                 print_mode=True):
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
        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

        self.topk_selection = args.topk_selection
        self.use_ratio_loss = args.use_ratio_loss
        self.use_token_dist_loss = args.use_token_dist_loss

        print('cls_weight=', clf_weight, 'distill_weight=', distill_weight)
        if not self.topk_selection and self.use_token_dist_loss:
            print('using KL divergence of final layer tokens in loss function (weight: see distill_weight above')
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
            pred, token_pred = outputs
        else:
            # predictions, final layer tokens (without CLS), final token keep decisions, all token keep decisions
            pred, token_pred, mask, out_pred_score = outputs


        # crossentropy loss between student predictions and ground truth labels
        cls_loss = self.base_criterion(pred, labels)

        with torch.no_grad():
            cls_t, token_t = self.teacher_model(inputs)  # 3rd return value is final CLS token attention weights

        # KL divergence between student class probabilities and teacher class probabilities
        cls_kl_loss = F.kl_div(
            F.log_softmax(pred, dim=-1),
            F.log_softmax(cls_t, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        if self.topk_selection:
            #token_kl_loss = F.kl_div(
            #    F.log_softmax(token_pred, dim=-1),
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
                for i, score in enumerate(out_pred_score):
                    if self.dynamic:
                        pos_ratio = score.mean()
                    else:
                        pos_ratio = score.mean(1)
                    pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

            if self.use_token_dist_loss:
                # final token distillation loss
                B, N, C = token_pred.size()
                assert mask.numel() == B * N
                bool_mask = mask.reshape(B * N) > 0.5
                token_pred = token_pred.reshape(B * N, C)
                token_t = token_t.reshape(B * N, C)
                if mask.sum() < 0.1:
                    token_kl_loss = token_pred.new(1, ).fill_(0.0)
                else:
                    token_t = token_t[bool_mask]
                    token_pred = token_pred[bool_mask]
                    if self.mse_token:
                        token_kl_loss = torch.pow(token_pred - token_t, 2).mean()
                    else:
                        # KL divergence between final encoder layer tokens (CLS token excluded)
                        token_kl_loss = F.kl_div(
                            F.log_softmax(token_pred, dim=-1),
                            F.log_softmax(token_t, dim=-1),
                            reduction='batchmean',
                            log_target=True
                        )

            # print(cls_loss, pred_loss)
            loss = self.clf_weight * cls_loss + self.distill_weight * cls_kl_loss
            if self.use_ratio_loss:
                loss += self.ratio_weight * pred_loss / len(self.pruning_loc)
            if self.use_token_dist_loss:
                loss += self.distill_weight * token_kl_loss

        if self.print_mode:
            self.cls_loss += cls_loss.detach().item()
            self.cls_distill_loss += cls_kl_loss.detach().item()
            if not self.topk_selection and self.use_ratio_loss:
                self.ratio_loss += pred_loss.detach().item()
            if not self.topk_selection and self.use_token_dist_loss:
                self.token_distill_loss += token_kl_loss.detach().item()
            self.count += 1
            if self.count % 100 == 1:
                if self.topk_selection:
                    print(f'loss info: cls_loss={(self.cls_loss / self.count):.4f}, '
                          f'cls_kl={(self.cls_distill_loss / self.count):.4f}')
                else:
                    loss_info = f'loss info: cls_loss={(self.cls_loss / self.count):.4f},   ' \
                                f'cls_kl={(self.cls_distill_loss / self.count):.4f}'
                    if self.use_ratio_loss:
                        loss_info += f',   ratio loss={(self.ratio_loss/self.count):.4f}'
                    if self.use_token_dist_loss:
                        loss_info += f',   final_token_kl={(self.token_distill_loss/self.count):.4f}'
                    print(loss_info)
        return loss
