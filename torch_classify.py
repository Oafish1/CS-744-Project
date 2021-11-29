# python torch_classify.py data/KKI.nel
import argparse

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler


parser = argparse.ArgumentParser(description='Perform graph classification.')
parser.add_argument('filename', type=str, help='File containing the labeled graphs')
parser.add_argument(
    '--heterogeneous',
    dest='heterogeneous',
    action='store_true',
    help='Graphs are not homogeneous',
)
args = parser.parse_args()

fname = args.filename
homogeneous = not args.heterogeneous

# Create dataset
class Dataset(DGLDataset):
    def __init__(self, name):
        super().__init__(name=name)

    def process(self):
        self.graphs = []
        self.labels = []

        # Read graphs from file
        num_vertices = 0; edges_src = []; edges_dst = []; edges_val = []
        with open(fname, 'r') as f:
            for line in f:
                line = line[:-1]

                if line == '':
                    if homogeneous:
                        edges_src = torch.Tensor(edges_src).to(torch.int32)
                        edges_dst = torch.Tensor(edges_dst).to(torch.int32)
                        graph = dgl.graph((edges_src, edges_dst), num_nodes=num_vertices)
                        # ASDF: Adjust features
                        graph.ndata['feat'] = torch.ones((num_vertices)).to(torch.float32)
                        graph.edata['weight'] = torch.Tensor(edges_val).to(torch.float32)

                        # Prevent error (Needs testing)
                        graph = dgl.add_self_loop(graph)
                    else:
                        raise Exception('Heterogeneous graph support not implemented yet')
                    self.graphs.append(graph)
                    self.labels.append(value)
                    num_vertices = 0; edges_src = []; edges_dst = []; edges_val = []
                else:
                    line = line.split(' ')

                    if line[0] == 'n':
                        num_vertices += 1
                    elif line[0] == 'e':
                        edges_src.append(int(line[1]) - 1)
                        edges_dst.append(int(line[2]) - 1)
                        edges_val.append(float(line[3]))
                    elif line[0] == 'g':
                        name = line[1]
                    elif line[0] == 'x':
                        value = float(line[1])

        self.labels = torch.LongTensor(self.labels)
        self.classes = len(np.unique(self.labels))

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

dataset = Dataset('dataset')

# https://docs.dgl.ai/en/0.6.x/new-tutorial/5_graph_classification.html
num_total = len(dataset)
num_train = int(num_total * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_total))

train_dataloader = GraphDataLoader(
    dataset,
    sampler=train_sampler,
    batch_size=5,
    drop_last=False,
)
test_dataloader = GraphDataLoader(
    dataset,
    sampler=test_sampler,
    batch_size=5,
    drop_last=False,
)

it = iter(train_dataloader)
batch = next(it)
print(batch)

# Classify
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation='relu')
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, features):
        v = self.conv1(g, features)
        # v = nn.ReLU(v)
        v = self.conv2(g, v)
        g.ndata['v'] = v
        return dgl.mean_nodes(g, 'v')

# ASDF: Adjust size
model = GCN(10, 20, dataset.classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        # ASDF: Adjust features
        pred = model(batched_graph, batched_graph.ndata['feat'])
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['feat'])
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)
