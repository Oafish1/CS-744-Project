import argparse
from time import perf_counter

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from classes import NelDataset, SimpleClassifier


# Parse arguments
parser = argparse.ArgumentParser(description='Perform graph classification')
parser.add_argument('filename', type=str, help='File containing the labeled graphs')
parser.add_argument(
    '--heterogeneous',
    dest='heterogeneous',
    action='store_true',
    help='Graphs are not homogeneous',
)
parser.add_argument('--master-ip', dest='master_ip', type=str)
parser.add_argument('--master-port', dest='master_port', default=6585, type=int)
parser.add_argument('--num-nodes', dest='num_nodes', type=int)
parser.add_argument('--rank', dest='rank', type=int)
parser.add_argument(
    '--silent',
    dest='silent',
    action='store_true',
    help='Do not print while training.  Useful for timing',
)
parser.add_argument(
    '--skip',
    dest='skip',
    action='store_true',
    help='Skip gradient sync occasionally',
)
args = parser.parse_args()
MASTER_IP = args.master_ip
MASTER_PORT = args.master_port
NUM_NODES = args.num_nodes
RANK = args.rank

args = parser.parse_args()
fname = args.filename
homogeneous = not args.heterogeneous
do_skip = args.skip
verbose = not args.silent

# Distributed setup
print('Initializing...')
MAIN_RANK = 0
torch.distributed.init_process_group(
    backend='gloo',
    init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}',
    rank=RANK,
    world_size=NUM_NODES
)
WORLD_SIZE = torch.distributed.get_world_size()
LOCAL_SIZE = torch.cuda.device_count()

# Load datasets
dataset = NelDataset('dataset', fname, homogeneous)
train_dataset, test_dataset, _ = dgl.data.utils.split_dataset(dataset, frac_list=[.8, .2, 0])

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_dataloader = GraphDataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=5,
    drop_last=False,
)
test_dataloader = GraphDataLoader(
    test_dataset,
    batch_size=5,
    drop_last=False,
)

# Perform classification
model = SimpleClassifier(1, 10, 100, dataset.classes).cpu()
model = torch.nn.parallel.DistributedDataParallel(model)
model.train()
criterion = nn.CrossEntropyLoss()
# TODO: Change optimizer (dist.optim?)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Optimization
def skip(i):
    return not (i+1 % 10 == 0)

def backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

training_time = -perf_counter()
forward_time = backward_time = 0
for epoch in range(100):
    epoch_loss = 0
    for batched_graph, labels in train_dataloader:
        # Forward
        forward_time -= perf_counter()
        logits = model(batched_graph)
        loss = criterion(logits, labels)
        forward_time += perf_counter()

        # Backward
        backward_time -= perf_counter()
        if do_skip and skip(epoch):
            with model.no_sync():
                backward(optimizer, loss)
        else:
            backward(optimizer, loss)
        backward_time += perf_counter()

        epoch_loss += loss.detach().item()
    epoch_loss /= len(train_dataloader)
    if verbose and (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, loss: {epoch_loss}')
training_time += perf_counter()

model.eval()
num_correct = 0
num_tests = 0
loss = 0
for batched_graph, labels in test_dataloader:
    logits = model(batched_graph)
    num_correct += (logits.argmax(1) == labels).sum().item()
    num_tests += len(labels)
    loss += criterion(logits, labels).detach().item()
loss /= len(test_dataloader)

print(f'Test accuracy: {num_correct / num_tests}')
print(f'Test loss: {loss}')
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
print(f'Parameters: {sum(p.numel() for p in model.parameters())}')
print(f'Forward Time: {forward_time}')
print(f'Backward Time: {backward_time}')
print(f'Training Time: {training_time}')
print(f'Formatted: {forward_time},{backward_time},{training_time}')
