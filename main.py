import numpy as np
import random
import scipy as sp
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim.lr_scheduler as lr_scheduler

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor

#from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from utils import GATConv
import scipy.sparse
from sklearn.cluster import Birch

import networkx as nx

import argparse
import utils
import os
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
#torch.use_deterministic_algorithms(True)

def parse_args():
    args = argparse.ArgumentParser(description='DGCluster arguments.')
    args.add_argument('--dataset', type=str, default='cora')
    args.add_argument('--lam', type=float, default=0.8)
    args.add_argument('--alp', type=float, default=0.8)
    args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    args.add_argument('--epochs', type=int, default=300)
    #args.add_argument('--base_model', type=str, default='gat', choices=['gcn', 'gat', 'gin', 'sage'])
    args.add_argument('--seed', type=int, default=0)
    args = args.parse_args()
    return args


def load_dataset(dataset_name):
    if dataset_name == 'cora':
        dataset = Planetoid(root='data', name="Cora")
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='data', name="Citeseer")
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='data', name="PubMed")
    elif dataset_name == 'computers':
        dataset = Amazon(root='data', name='Computers')
    elif dataset_name == 'photo':
        dataset = Amazon(root='data', name='Photo')
    elif dataset_name == 'coauthorcs':
        dataset = Coauthor(root='data/Coauthor', name='CS')
    elif dataset_name == 'coauthorphysics':
        dataset = Coauthor(root='data/Coauthor', name='Physics')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} not implemented.')
    return dataset


class GNN(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(GNN, self).__init__()

        self.conv1 = GATConv(in_dim,64)
        self.conv2 = GATConv(64*5,32)
        self.conv3 = GATConv(32*5,16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = (F.hardtanh(x,min_val=0,max_val=1)) ** 2
        x = F.normalize(x)

        return x


def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj


def aux_objective(output, s):
    sample_size = len(s)

    out = output[s, :].float()

    C = oh_labels[s, :].float()

    X = C.sum(dim=0)
    X = X ** 2
    X = X.sum()

    Y = torch.matmul(torch.t(out), C)
    Y = torch.matmul(Y, torch.t(Y))
    Y = torch.trace(Y)

    t1 = torch.matmul(torch.t(C), C)
    t1 = torch.matmul(t1, t1)
    t1 = torch.trace(t1)

    t2 = torch.matmul(torch.t(out), out)
    t2 = torch.matmul(t2, t2)
    t2 = torch.trace(t2)

    t3 = torch.matmul(torch.t(out), C)
    t3 = torch.matmul(t3, torch.t(t3))
    t3 = torch.trace(t3)

    aux_objective_loss = 1 / (sample_size ** 2) * (t1 + t2 - 2 * t3)

    return aux_objective_loss


def conduc_loss(output,s):
    x, edge_index = data.x, data.edge_index
    #edge_index = data.edge_index
    # Create self-loop edges
    self_loops = torch.arange(0, num_nodes, device=device).unsqueeze(0).repeat(2, 1)

    # Combine self-loops with existing edges
    edge_index_with_loops = torch.cat([data.edge_index, self_loops], dim=1)

    # Create the sparse adjacency tensor with self-loops
    torch_sparse_adj_self = torch.sparse_coo_tensor(
        edge_index_with_loops,
        torch.ones(edge_index_with_loops.size(1), device=device),
        size=(num_nodes, num_nodes)
    )
    degrees = torch_sparse_adj_self.sum(dim=1).to_dense()  # Dense tensor of degrees
    D_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))  # D^(-1/2)

    # Convert sparse adjacency to dense for normalization
    dense_adj = torch_sparse_adj_self.to_dense()

    # Symmetrically normalize the adjacency matrix
    normalized_adj = D_inv_sqrt @ dense_adj @ D_inv_sqrt
    normalized_adj = torch.nan_to_num(normalized_adj, nan=0.0)

    A2 = torch.matmul(normalized_adj, normalized_adj)
    A3 = torch.matmul(normalized_adj, A2)
    A4 = torch.matmul(normalized_adj, A3)
    A_hop = normalized_adj+A2+A3+A4
    s_output = output[s, :]
    s_adj = A_hop[s, :][:, s]
    s_adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=s_adj[edge_index[0], edge_index[1]],
        size=(num_nodes, num_nodes),
    )  
   
    x = torch.matmul(torch.t(s_output).double(),s_adj.double())
    x = torch.trace(torch.matmul(x,s_output.double()))  
    weights = torch.sum(s_adj)/2
    return (1-x/weights)



def loss_fn(output, lam=0.0, alp=0.0, epoch=-1):
    sample_size = int(1 * num_nodes)
    s = random.sample(range(0, num_nodes), sample_size)

    s_output = output[s, :]

    s_adj = sparse_adj[s, :][:, s]
    s_adj = convert_scipy_torch_sp(s_adj)
    s_degree = degree[s]

    x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))
    x = torch.matmul(x, s_output.double())
    x = torch.trace(x)

    y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
    y = (y ** 2).sum()
    y = y / (2 * num_edges)

    # scaling=1
    scaling = num_nodes ** 2 / (sample_size ** 2)

    m_loss = -((x - y) / (2 * num_edges)) * scaling

    aux_loss = lam * aux_objective(output, s)

    conductance_loss = conduc_loss(output,s)*alp
    loss = m_loss + aux_loss + conductance_loss
    if epoch % 10 ==0:
       print('epoch: ', epoch, 'loss: ', loss.item(), 'm_loss: ', m_loss.item(), 'aux_loss: ', aux_loss.item(), 'reg_loss: ', conductance_loss.item())

    return loss


def train(model, optimizer, data, epochs, lam, alp):
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)

        loss = loss_fn(out, lam, alp, epoch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    lam = args.lam
    alp = args.alp
    epochs = args.epochs
    device = args.device

    seeds = [88,96,17,75,15,69,5,115,784,447]
    mods = []
    nmis = []
    f1s=[]
    conds=[]    
    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # device selection
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device(device)
        else:
            device = torch.device('cpu')
        print(f'Using device: {device}')

        # transform data
        transform = T.NormalizeFeatures()

        # load dataset
        dataset = load_dataset(dataset_name)
        data = dataset[0]
        data = data.to(device)

        # preprocessing
        num_nodes = data.x.shape[0]
        num_edges = (data.edge_index.shape[1])
        labels = data.y.flatten()
        oh_labels = F.one_hot(labels, num_classes=max(labels) + 1)

        sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), data.edge_index.cpu().numpy()), shape=(num_nodes, num_nodes))
        torch_sparse_adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(num_edges).to(device), size=(num_nodes, num_nodes))
        degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(device)
        Graph = nx.from_scipy_sparse_array(sparse_adj, create_using=nx.Graph).to_undirected()
        num_edges = int((data.edge_index.shape[1]) / 2)

        in_dim = data.x.shape[1]
        out_dim = 64
        model = GNN(in_dim, out_dim).to(device)

        optimizer_name = "Adam"
        lr = 1e-3
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001, amsgrad=True)

        train(model, optimizer, data, epochs, lam, alp)

        test_data = data.clone()
        print(test_data)

        model.eval()
        x = model(test_data)

        clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(x.detach().cpu().numpy(), y=None)
        FQ = utils.compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device)
        print('No of clusters: ', max(clusters) + 1)
        print('Modularity:', FQ)

        NMI = utils.compute_nmi(clusters, data.y.squeeze().cpu().numpy())
        print('NMI:', NMI)

        conductance = utils.compute_conductance(clusters, Graph)
        avg_conductance = sum(conductance) / len(conductance)
        print(avg_conductance * 100)

        f1_score = utils.sample_f1_score(test_data, clusters, num_nodes)
        print('Sample_F1_score:', f1_score)

        try:
            avg_conductance = sum(conductance) / len(conductance)
            print(avg_conductance * 100)
            conds.append(avg_conductance)
            nmis.append(NMI)
            mods.append(FQ)       
            f1s.append(f1_score)
        except ZeroDivisionError:
            continue   
        results = {
            'num_clusters': np.unique(clusters).shape[0],
            'modularity': FQ,
            'nmi': NMI,
            'conductance': avg_conductance,
            'sample_f1_score': f1_score
        }

        if not os.path.exists('results'):
            os.makedirs('results')
        if alp == 0.0:
            torch.save(results, f'results/results_{dataset_name}_{lam}_{epochs}_{seed}.pt')
        else:
            torch.save(results, f'results/results_{dataset_name}_{lam}_{alp}_{epochs}_{seed}.pt')
    print("Avearge modularity",round(statistics.mean(mods),4))
    print("Average NMI",round(statistics.mean(nmis),4))
    print("Average Conductance",round(statistics.mean(conds),4))
    print("Average F1 Score",round(statistics.mean(f1s),4))
