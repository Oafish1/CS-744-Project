# python torch_classify.py data/KKI.nel
# python torch_classify.py data/OHSU.nel
# python torch_classify.py data/Peking_1.nel
import argparse

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
                        # graph.ndata['feat'] = torch.ones((num_vertices)).to(torch.float32)
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
                        value = int((1 + float(line[1])) / 2)

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

# Classify
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        h_mean = dgl.mean_nodes(g, 'h')
        return torch.sigmoid(self.linear(h_mean))

model = Classifier(1, 10, dataset.classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(100):
    epoch_loss = 0
    for batched_graph, labels in train_dataloader:
        logits = model(batched_graph)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= len(train_dataloader)
    if (epoch+1) % 10 == 0:
        print(f'Epoch:{epoch+1}, loss:{epoch_loss}')

num_correct = 0
num_tests = 0
loss = 0
for batched_graph, labels in test_dataloader:
    logits = model(batched_graph)
    num_correct += (logits.argmax(1) == labels).sum().item()
    num_tests += len(labels)
    loss += criterion(logits, labels).detach().item()
loss /= len(test_dataloader)

print(f'Test accuracy:{num_correct / num_tests}')
print(f'Test loss:{loss}')
