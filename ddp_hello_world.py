import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

import time

import torchvision
from torchvision import transforms

from torch.nn.parallel import DistributedDataParallel as DDP

import utils

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def get_model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return Net()

def get_dataloaders(rank, world_size, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/scratch_net/biwidl215/segerm/CIFAR10', train=True,
                                            download=True, transform=transform)
    trainsampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=False,
                                              num_workers=0, drop_last=False, shuffle=False, sampler=trainsampler)

    testset = torchvision.datasets.CIFAR10(root='/scratch_net/biwidl215/segerm/CIFAR10', train=False,
                                           download=True, transform=transform)
    testsampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=False,
                                             num_workers=0, drop_last=False, shuffle=False, sampler=testsampler)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, args):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, args.num_gpus)

    # create model and move it to GPU with id rank
    model = get_model().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train_loader, test_loader = get_dataloaders(rank, args.num_gpus, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(f'Starting training in process {rank} with {len(train_loader)*args.batch_size} training samples, and '
          f'{len(test_loader)*args.batch_size} validation samples')
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #if i % 20 == 19:  # print every 2000 mini-batches
            #    print(f'Rank{rank}: [{epoch + 1}, {i + 1}] loss: {(running_loss/20):.4f}')
            #    running_loss = 0.0

    print(f'Finished Training on process {rank}')

    cleanup()

def train(device, args):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/scratch_net/biwidl215/segerm/CIFAR10', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, pin_memory=False,
                                              num_workers=0, drop_last=False, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='/scratch_net/biwidl215/segerm/CIFAR10', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, pin_memory=False,
                                             num_workers=0, drop_last=False, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # create model and move it to GPU with id rank
    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #if i % 20 == 19:  # print every 2000 mini-batches
            #    print(f'Rank{rank}: [{epoch + 1}, {i + 1}] loss: {(running_loss/20):.4f}')
            #    running_loss = 0.0

    print(f'Finished Training')


def run_demo(demo_fn, args):
    print(f'Spawning {args.num_gpus} processes running {demo_fn.__name__}')
    mp.spawn(demo_fn,
             args=(args,),
             nprocs=args.num_gpus,
             join=True)

if __name__ == '__main__':
    start = time.time()
    args = utils.parse_args()
    if args.use_ddp:
        args.num_gpus = torch.cuda.device_count()
        print(f'World size = {args.num_gpus}')
        run_demo(demo_basic, args)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train(device, args)
    print(f'Finished training, took {time.time()-start} seconds')