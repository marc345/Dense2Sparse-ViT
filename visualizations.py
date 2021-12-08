def visualize(model, teacher_model, current_epoch, test_imgs, test_labels, avg_cls_attn_list):
    model.eval()
    with torch.no_grad():

        if not args.visualize_cls_attn_evo and not args.visualize_patch_drop:
            return

        # attention_segmentation.cls_attention_histogram(args, current_epoch+1, avg_cls_attn_list)

        if not args.random_drop and args.cls_from_teacher:
            cls_attn_weights = teacher_model.forward_cls_attention(test_imgs.clone())  # (B, L, H, N+1)
            test_logits, cls_attns, pred_logits, final_policy = model(test_imgs.clone(), cls_attn_weights)
        else:
            test_logits, cls_attns, pred_logits, final_policy = model(test_imgs.clone())

        test_preds = torch.argmax(test_logits, dim=1)

        kept_token_idx = getattr(model, "kept_token_indices")
        dropped_token_idx = getattr(model, "dropped_token_indices")
        token_idx = torch.cat((kept_token_idx, dropped_token_idx), dim=1)

        keep_mask = torch.ones_like(kept_token_idx)
        drop_mask = torch.zeros_like(dropped_token_idx)
        sorted_patch_drop_mask = torch.cat((keep_mask, drop_mask), dim=1)
        patch_drop_mask = torch.empty_like(sorted_patch_drop_mask)
        patch_drop_mask.scatter_(dim=1, index=token_idx.long(), src=sorted_patch_drop_mask).unsqueeze(-1)

        # only display result after last predictor stage
        if args.visualize_patch_drop:
            attention_segmentation.display_patch_drop(test_imgs.cpu(), patch_drop_mask.cpu(), args, current_epoch + 1,
                                                      (test_preds == test_labels).cpu().numpy(),
                                                      patch_indices=[kept_token_idx.cpu(), dropped_token_idx.cpu()],
                                                      patch_scores=patch_keep_prob.cpu() if not args.topk_selection
                                                      else None)

        if args.visualize_cls_attn_evo:
            padded_cls_attns = []
            for i, attn in enumerate(cls_attns):
                N = int((mask_test_imgs.shape[-1] // args.patch_size) ** 2)
                if i < args.pruning_locs[0]:
                    B, H, N = attn[:, :, 1:].shape
                    padded_cls_attns.append(attn.unsqueeze(1))
                else:
                    B, H, N_kept = attn[:, :, 1:].shape
                    padded_attn = torch.cat((attn, torch.zeros((B, H, N - N_kept),
                                                               device=attn.device, dtype=attn.dtype)), dim=2)
                    padded_cls_attns.append(padded_attn.unsqueeze(1))
            # concatenate the list of class attentions after each encoder layer
            # permute layer and batch dimension, such that we can visualize the evolution of the CLS token for the same
            # image across all layers in one picture and loop over the batch dimension to plot this picture for every
            # input image in the batch
            cls_attns = torch.cat(padded_cls_attns, dim=1)  # (B, L, H, N+1)
            for b in range(cls_attns.shape[0]):
                attention_segmentation.visualize_heads(test_imgs[b].cpu(), args, current_epoch + 1,
                                                       [kept_token_idx.cpu(), dropped_token_idx.cpu()],
                                                       cls_attns[b].cpu(), b)