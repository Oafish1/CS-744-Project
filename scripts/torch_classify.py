import argparse
from time import perf_counter

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
import numpy as np
import torch
from torch import nn
import torch.distributed as dd
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
parser.add_argument(
    '--gs',
    dest='gather_scatter',
    action='store_true',
    help='Use gather-scatter all-reduce',
)
parser.add_argument(
    '--rr',
    dest='ring_reduce',
    action='store_true',
    help='Use ring-reduce',
)
parser.add_argument(
    '--hy',
    dest='hybrid_reduce',
    action='store_true',
    help='Use hybrid-reduce',
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
gather_scatter = args.gather_scatter
ring_reduce = args.ring_reduce
hybrid_reduce = args.hybrid_reduce
assert not (gather_scatter + ring_reduce + hybrid_reduce > 1), (
    'Only one of ``gather_scatter``, ``ring_reduce``, or ``hybrid_reduce`` may be used.'
)
integrated_model = not (gather_scatter or ring_reduce or hybrid_reduce)


# Distributed setup
print('Initializing...')
MAIN_RANK = 0
dd.init_process_group(
    backend='gloo',
    init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}',
    rank=RANK,
    world_size=NUM_NODES
)
WORLD_SIZE = dd.get_world_size()
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
model = SimpleClassifier(1, 10, 10, dataset.classes).cpu()
if integrated_model:
    model = torch.nn.parallel.DistributedDataParallel(model)
model.train()
criterion = nn.CrossEntropyLoss()
# TODO: Change optimizer (dist.optim?)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Skip sync
def skip(i):
    to_sync = ((i+1) % 10 == 0)
    return not to_sync

def backward(optimizer, loss, model=None, skip_sync=True, epoch=None):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO: Potentially use LOCAL_SIZE in all calculations
    # or implement hierarchical sync
    if (hybrid_reduce or not skip_sync) and WORLD_SIZE > 1:
        # Could be optimized by not doing for each individually
        for param in model.parameters():
            grad = param.grad

            # Gather-Scatter
            if gather_scatter:
                if dd.get_rank() == MAIN_RANK:
                    # Gather
                    grad_list = [torch.zeros_like(grad) for _ in range(WORLD_SIZE)]
                    dd.gather(grad, gather_list=grad_list, dst=MAIN_RANK)

                    # Compute average
                    avg_grad = torch.stack(grad_list).mean(0)
                    grad_list = [avg_grad for _ in range(WORLD_SIZE)]

                    # Scatter
                    dd.scatter(grad, scatter_list=grad_list, src=MAIN_RANK)
                else:
                    dd.gather(grad, dst=MAIN_RANK)
                    dd.scatter(grad, src=MAIN_RANK)

            elif ring_reduce:
                rank = dd.get_rank()
                prev_rank = (rank-1) % WORLD_SIZE
                next_rank = (rank+1) % WORLD_SIZE

                # Get from prev
                if not rank == MAIN_RANK:
                    new_grad = torch.zeros_like(grad)
                    dd.recv(new_grad, prev_rank)
                    # Sum
                    sum_grad = grad + new_grad
                else:
                    sum_grad = grad

                # Send to next
                dd.send(sum_grad, next_rank)

                # Average and distribute
                dd.recv(grad, prev_rank)
                if rank == MAIN_RANK:
                    grad /= WORLD_SIZE
                if next_rank != MAIN_RANK:
                    dd.send(grad, next_rank)

            elif hybrid_reduce and dd.get_rank() in (epoch % WORLD_SIZE, (epoch+1) % WORLD_SIZE):
                # TODO: Potentially use LOCAL_SIZE
                rank = dd.get_rank()
                send_rank = epoch % WORLD_SIZE
                recv_rank = (epoch+1) % WORLD_SIZE
                if rank == send_rank:
                    dd.send(grad, recv_rank)
                else:
                    new_grad = torch.zeros_like(grad)
                    dd.recv(new_grad, send_rank)
                    grad += new_grad
                    grad /= 2
                # TODO: Add sync on last epoch


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
        # TODO: Sync fewer times, maybe only at the end of the epoch
        backward_time -= perf_counter()
        if do_skip and skip(epoch) and integrated_model:
            with model.no_sync():
                backward(optimizer, loss)
        else:
            backward(optimizer, loss, model=model, skip_sync=(skip(epoch) or integrated_model), epoch=epoch)
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
