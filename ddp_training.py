### DDP Functions

# setup process group for DistributedDataParallel
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f'Setting up process group {rank}')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, data_folder, batch_size=64, pin_memory=False, num_workers=0):
    ddp_data_sets = {x: datasets.ImageFolder(data_folder+f'/{x}', transform=data_transforms[x])
                for x in ['train', 'val']}

    # Sampler that restricts data loading to a subset of the dataset.
    ddp_samplers = {x: DistributedSampler(ddp_data_sets[x], num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
               for x in ['train', 'val']}

    # Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    ddp_data_loaders = {x: DataLoader(ddp_data_sets[x], batch_size=batch_size, pin_memory=pin_memory,
                                      num_workers=num_workers, drop_last=False, shuffle=False, sampler=ddp_samplers[x])
                        for x in ['train', 'val']}

    return ddp_data_loaders

def cleanup():
    dist.destroy_process_group()

######################################################################
# Training the model using DistributedDataParallel
# ------------------


def train_model_ddp(rank, args):
    print(f'Entering distributed data parallel training function in process {rank}')
    try:
        # setup the process groups
        setup(rank, args.world_size)
        print(f'Spawned distributed data parallel training process {rank}')
        since = time.time()

        best_acc = 0.0

        #  overfit on single training data batch test
        #data = {phase: next(iter(ddp_data_loaders[phase])) for phase in ['train', 'val']}
        #batch_repeat_factor = 20

        # prepare the dataloaders
        ddp_data_loaders = prepare(rank, args.world_size, args.imgnet_val_dir, batch_size=args.batch_size)

        if rank == 0:
            mask_test_data = next(iter(ddp_data_loaders['val']))
            mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
            mask_test_imgs = mask_test_imgs.to(rank)
            mask_test_labels = mask_test_labels.to(rank)

            # Writer will output to ./runs/ directory by default
            writer = SummaryWriter(log_dir=f'runs/{os.environ["SLURM_JOBID"]}')

        # get the model specified as argument
        student = utils.get_model({'model_name': 'dynamic_vit_student', 'patch_size': 16}, pretrained=True)
        teacher = utils.get_model({'model_name': 'dynamic_vit_teacher', 'patch_size': 16}, pretrained=True)
        teacher = teacher.to(rank)
        teacher.eval()

        for param in teacher.parameters():
            param.requires_grad = False

        # params_to_optimize = []
        # for name, param in student.named_parameters():
        #    if 'predictor' in name:
        #        param.requires_grad_(True)
        #        params_to_optimize.append(param)
        #    else:
        #        param.requires_grad_(False)

        parameter_group = utils.get_param_groups(student, args.weight_decay)

        opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(parameter_group, **opt_args)

        criterion = losses.DistillDiffPruningLoss(teacher_model=teacher, clf_weight=1.0,
                                                  base_criterion=torch.nn.CrossEntropyLoss(), print_mode=False)

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        # move model to the right device/process running on that device
        student = student.to(rank)

        # device_ids tell DDP where is your model
        # output_device tells DDP where to output, in our case, it is rank
        # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
        ddp_student = DDP(student, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        print(f'Starting training on rank {rank}, with {len(ddp_data_loaders["train"])*args.batch_size} training samples '
              f'and {len(ddp_data_loaders["val"])*args.batch_size} validation samples for {args.epochs} epochs')

        for epoch in range(args.epochs):
            print('Epoch {}/{}'.format(epoch+1, args.epochs))
            print('-' * 10)

            for phase in ['train']:

                if phase == 'train':
                    warmup_step = 5
                    utils.adjust_learning_rate(optimizer.param_groups, args.lr, args.min_lr, epoch, args.epochs,
                                               warmup_predictor=False, warming_up_step=warmup_step, base_multi=0.1)

                running_loss = torch.zeros(1, dtype=torch.float32).to(rank)
                running_corrects = torch.zeros(1, dtype=torch.int32).to(rank)
                running_keeping_ratio = torch.zeros(1, dtype=torch.float32).to(rank)
                num_samples = torch.zeros(1, dtype=torch.int32).to(rank)

                ddp_student.train(mode=(phase == 'train'))

                for i, data in enumerate(tqdm(ddp_data_loaders[phase])):
                #for i in tqdm(range(batch_repeat_factor)):

                    ddp_data_loaders[phase].sampler.set_epoch(epoch)

                    inputs = data[0].to(rank)
                    labels = data[1].to(rank)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            #with torch.cuda.amp.autocast():
                            output = ddp_student(inputs.clone())
                            loss = criterion(inputs, output, labels)
                            ## this attribute is added by timm on one optimizer (adahessian)
                            #is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                            #loss_scaler(loss, optimizer, clip_grad=max_norm,
                            #            parameters=model.parameters(), create_graph=is_second_order)
                            # backward + optimize only if in training phase

                            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                            # Backward passes under autocast are not recommended.
                            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                            #scaler.scale(loss).backward()
                            loss.backward()

                            # scaler.step() first unscales the gradients of the optimizer's assigned params.
                            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                            # otherwise, optimizer.step() is skipped.
                            #scaler.step(optimizer)

                            # Updates the scale for next iteration.
                            #scaler.update()
                            optimizer.step()
                            preds = torch.argmax(output[0].detach(), dim=1)
                        else:
                            output = ddp_student(inputs.clone())
                            loss = F.cross_entropy(output, labels)
                            preds = torch.argmax(output.detach(), dim=1)


                    # statistics
                    running_loss += loss
                    running_corrects += torch.sum(preds == labels.data)
                    num_samples += inputs.shape[0]
                    for i, decision in enumerate(ddp_student.module.decisions):
                        if phase == 'train':
                            # mean token keeping ratio across batch
                            running_keeping_ratio += torch.mean(torch.sum(decision, dim=1)/decision.shape[1])
                        else:
                            running_keeping_ratio += ddp_student.module.token_ratio[-1]

                #if phase == 'train':
                #    scheduler.step(epoch)
                # reduce metrics to rank 0 (main node)
                dist.reduce(running_loss, 0, op=dist.ReduceOp.SUM)
                dist.reduce(running_corrects, 0, op=dist.ReduceOp.SUM)
                dist.reduce(running_keeping_ratio, 0, op=dist.ReduceOp.SUM)
                dist.reduce(num_samples, 0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    total_loss = (running_loss / num_samples).item()
                    total_acc = (running_corrects / num_samples).item()
                    total_keeping_ratio = (running_keeping_ratio / num_samples).item()
                    print(f'Rank {rank}, {phase} loss: {total_loss:.4f}, acc: {total_acc:.4f}, '
                          f'kept token ratio: {total_keeping_ratio:.4f}')

                    # Tensorboard tracking
                    writer.add_scalar(f'{phase}_metrics/total_loss', total_loss, epoch)
                    writer.add_scalar(f'{phase}_metrics/total_acc', total_acc, epoch)
                    writer.add_scalar(f'{phase}_metrics/kept_token_ratio', total_keeping_ratio, epoch)

            if rank == 0:
                with torch.no_grad():
                    student.eval()
                    test_outs = student(mask_test_imgs.clone())
                    test_preds = torch.argmax(test_outs[0], dim=1)

                    for i, decision in enumerate(ddp_student.module.decisions):
                        num_keep_tokens = int(decision.shape[1] * ddp_student.module.token_ratio[i])
                        kept_token_idx = decision[:, :num_keep_tokens]
                        dropped_token_idx = decision[:, num_keep_tokens:]
                        keep_mask = torch.ones_like(kept_token_idx).to(rank)
                        drop_mask = torch.zeros_like(dropped_token_idx).to(rank)
                        patch_drop_mask = torch.cat((keep_mask, drop_mask), dim=1)
                        patch_drop_mask = torch.gather(patch_drop_mask, index=decision.long(), dim=1)
                        dummy_decisison = torch.zeros(patch_drop_mask.shape[0], 2, patch_drop_mask.shape[2]).to(rank)
                        patch_drop_mask = torch.cat((dummy_decisison, patch_drop_mask), dim=1)

                    # only display result after last predictor stage
                    attention_segmentation.display_patch_drop(mask_test_imgs.cpu(), patch_drop_mask.cpu(),
                                                              "test_imgs/mask_predictor", epoch,
                                                              (test_preds == mask_test_labels).cpu().numpy(),
                                                              display_segmentation=False, max_heads=True)

            dist.barrier()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    finally:
        cleanup()

########################################################################