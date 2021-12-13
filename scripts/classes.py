import dgl
from dgl.data import DGLDataset
from dgl.nn import GraphConv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NelDataset(DGLDataset):
    def __init__(self, name, fname, homogeneous=True):
        self.fname = fname
        self.homogeneous = homogeneous
        super().__init__(name=name)

    def process(self):
        self.graphs = []
        self.labels = []

        # Read graphs from file
        num_vertices = 0; edges_src = []; edges_dst = []; edges_val = []
        with open(self.fname, 'r') as f:
            for line in f:
                line = line[:-1]

                if line == '':
                    if self.homogeneous:
                        edges_src = torch.Tensor(edges_src).to(torch.int32)
                        edges_dst = torch.Tensor(edges_dst).to(torch.int32)
                        graph = dgl.graph((edges_src, edges_dst), num_nodes=num_vertices)
                        graph.edata['weight'] = torch.Tensor(edges_val).to(torch.float32)

                        # Add self-loops to prevent disconnect error
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


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, num_hidden_dim, hidden_dim, out_dim):
        super().__init__()
        current_dim = in_dim
        self.conv = nn.ModuleList()
        for _ in range(num_hidden_dim):
            self.conv.append(GraphConv(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.linear = nn.Linear(current_dim, out_dim)

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.conv:
            h = F.relu(conv(g, h))
        g.ndata['h'] = h
        h_mean = dgl.mean_nodes(g, 'h')
        return torch.sigmoid(self.linear(h_mean))
