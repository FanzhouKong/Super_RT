
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import from_smiles, to_networkx
import torch
from rdkit import Chem
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from typing import Optional
from torch_geometric.data import Data
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import dgl
from math import sqrt
# import deepchem as dc

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks



def make_combined_df(mg, top_features):
    df = mg.copy()
    if len(mg)!= len(top_features):
        print('the dataframe has different legnth, please double check')
        return np.NAN
    for i in range(len(mg)):
        df[i]['md']=torch.tensor(top_features.iloc[i].values, dtype=np.float32).view(1, -1)
    return df
# def make_dc_graphlist(dc_dataset, smiles_col = 'smiles', y_col = 'rt'):
#     featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
#     feats = featurizer.featurize(dc_dataset[smiles_col], log_every_n=10)
#     dataset = dc.data.NumpyDataset(feats, dc_dataset[y_col].values, 
#                                    ids = dc_dataset['smiles'].values
#                                    )
#     dataset = dc_to_graphlist(dataset)
    return dataset
# def dc_to_graphlist(dc_dataset):
#     temp = dc_dataset.to_dataframe()
#     graph_list = []
#     for i in range(len(temp)):
#         data_row = Data(x = torch.tensor(temp.X[i].node_features).to(torch.float32), 
#                         edge_index=torch.tensor(temp.X[i].edge_index).to(torch.int64), 
#                         edge_attr=torch.tensor(temp.X[i].edge_features).to(torch.int64), 
#                         y = torch.tensor(temp.y[i], dtype=torch.float).view(1, -1),
#                         smiles = temp.ids[i]
#                         )
#         graph_list.append(data_row)
#     return(graph_list)
# def make_modred_df(data, smile_col = 'smiles', y_col = 'rt'):
#     f2 = dc.feat.MordredDescriptors(ignore_3D=True)
#     feats2 = f2.featurize(data[smile_col])
#     descriptors_df = pd.DataFrame(feats2)
#     descriptors_df.columns = [f'md_{i+1}' for i in range(descriptors_df.shape[1])]
#     descriptors_df.insert(0, 'y', data[y_col].values)
#     return descriptors_df
# Optionally, you can set column names if needed
    descriptors_df.columns = [f'md_{i+1}' for i in range(descriptors_df.shape[1])]
    descriptors_df.insert(0, 'y', data[y_col].values)
    return descriptors_df
# def rdkit_featurzier(smiles, use_BCUT = False):
#     all_descriptors = {name: func for name, func in Descriptors.descList}
#     all_names = list((all_descriptors.keys()))
#     if use_BCUT == False:
#         all_names = [x for x in all_names if x.startswith('BCUT2D_')== False]
#     f2 = dc.feat.RDKitDescriptors(use_bcut2d= use_BCUT)
#     feats2 = f2.featurize(smiles)
#     return(feats2[0])
# def make_rdkit_df(data, smile_col = 'smiles', y_col = 'rt', use_BCUT = False):
#     all_descriptors = {name: func for name, func in Descriptors.descList}
#     all_names = list((all_descriptors.keys()))
#     if use_BCUT == False:
#         all_names = [x for x in all_names if x.startswith('BCUT2D_')== False]
    
#     f2 = dc.feat.RDKitDescriptors(use_bcut2d= use_BCUT)
#     feats2 = f2.featurize(data[smile_col])
#     descriptors_df = pd.DataFrame(feats2)
# Optionally, you can set column names if needed
    descriptors_df.columns = all_names
    descriptors_df.insert(0, 'y', data[y_col].values)
    return descriptors_df
def train(train_loader,device, optimizer, model):
    # print('tt')
    total_loss = total_samples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_samples += data.num_graphs
    return sqrt(total_loss / total_samples)
@torch.no_grad()
def test(loader,device, model):
    mse = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr,data.batch)
            l = F.mse_loss(out, data.y, reduction='none').cpu()
            mse.append(l)
        rmse = float(torch.cat(mse, dim=0).mean().sqrt())
    return rmse


class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        # propagate_type: (x: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)
# class AttentiveFP_dev(torch.nn.Module):

#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         out_channels: int,
#         edge_dim: int,
#         num_layers: int,
#         num_timesteps: int,
#         num_md:int,
#         dropout: float = 0.0,
#     ):
#         super(AttentiveFP_dev, self).__init__()

#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.edge_dim = edge_dim
#         self.num_layers = num_layers
#         self.num_timesteps = num_timesteps
#         self.dropout = dropout

#         self.lin1 = Linear(in_channels, hidden_channels)

#         self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
#                                   dropout)
#         self.gru = GRUCell(hidden_channels, hidden_channels)

#         self.atom_convs = torch.nn.ModuleList()
#         self.atom_grus = torch.nn.ModuleList()
#         for _ in range(num_layers - 1):
#             conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
#                            add_self_loops=False, negative_slope=0.01)
#             self.atom_convs.append(conv)
#             self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

#         self.mol_conv = GATConv(hidden_channels, hidden_channels,
#                                 dropout=dropout, add_self_loops=False,
#                                 negative_slope=0.01)
#         self.mol_conv.explain = False  # Cannot explain global pooling.
#         self.mol_gru = GRUCell(hidden_channels, hidden_channels)

#         self.lin2 = Linear(hidden_channels, out_channels)
#         self.lin3 = Linear(num_md+1, 1)

#         self.reset_parameters()

#     def reset_parameters(self):
#         r"""Resets all learnable parameters of the module."""
#         self.lin1.reset_parameters()
#         self.gate_conv.reset_parameters()
#         self.gru.reset_parameters()
#         for conv, gru in zip(self.atom_convs, self.atom_grus):
#             conv.reset_parameters()
#             gru.reset_parameters()
#         self.mol_conv.reset_parameters()
#         self.mol_gru.reset_parameters()
#         self.lin2.reset_parameters()
#         self.lin3.reset_parameters()

#     def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
#                 batch: Tensor, md: Tensor) -> Tensor:
#         """"""  # noqa: D419
#         # Atom Embedding:
#         x = F.leaky_relu_(self.lin1(x))

#         h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         x = self.gru(h, x).relu_()

#         for conv, gru in zip(self.atom_convs, self.atom_grus):
#             h = conv(x, edge_index)
#             h = F.elu(h)
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             x = gru(h, x).relu()

#         # Molecule Embedding:
#         row = torch.arange(batch.size(0), device=batch.device)
#         edge_index = torch.stack([row, batch], dim=0)

#         out = global_add_pool(x, batch).relu_()
#         for t in range(self.num_timesteps):
#             h = F.elu_(self.mol_conv((x, out), edge_index))
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             out = self.mol_gru(h, out).relu_()

#         # Predictor:
#         out = F.dropout(out, p=self.dropout, training=self.training)
#         out = self.lin2(out)
#         out = torch.cat([out, md], dim=1)
#         out = self.lin3(out)
#         # for t in range(len(out)):
#         #     out[t] = torch.cat([out[t], ], dim=1)
        
#         return out
class AttentiveFP_super(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        descriptor_channels:int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super(AttentiveFP_super, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.descriptor_channels = descriptor_channels
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        input_fc_hidden1 = 2*descriptor_channels
        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        self.mol_gru.fc = nn.Identity()
        self.meta_fc = nn.Sequential(
            nn.Linear(in_features=descriptor_channels, out_features=input_fc_hidden1),
            nn.BatchNorm1d(input_fc_hidden1),
            nn.ReLU(),
            nn.Linear(input_fc_hidden1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_channels + 128, 128),  # Adjust based on the features from resnet
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for regression
        )
        # self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        # self.lin2.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,descriptors:Tensor,
                batch: Tensor) -> Tensor:
        """"""  # noqa: D419
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        out = F.dropout(out, p=self.dropout, training=self.training)
        descriptor_features = self.meta_fc(descriptors)
  
        combined_features = torch.cat([descriptor_features,out], dim=1)
        # Predictor:
        output = self.combined_fc(combined_features)

        return output
class AttentiveFP_super_2(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        descriptor_channels:int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super(AttentiveFP_super_2, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        self.mol_gru.fc = nn.Identity()
        self.lin2 = Linear(hidden_channels, out_channels)
        # above is original code
        self.meta_fc = nn.Sequential(
            nn.Linear(in_features=descriptor_channels, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU() 
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(out_channels + 64, 256),  # Adjust based on the features from resnet
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for regression
        )


        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()
        # self.meta_fc.reset_parameters()
        # self.combined_fc.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,descriptors:Tensor,
                batch: Tensor) -> Tensor:
        """"""  # noqa: D419
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)
        descriptor_features = self.meta_fc(descriptors)
        combined_features = torch.cat([out, descriptor_features], dim=1)
        # Predictor:
        output = self.combined_fc(combined_features)

        return output
class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        """"""  # noqa: D419
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)
class AttentiveFP_alt(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            edge_dim: int,
            num_layers: int,
            num_timesteps: int,
            num_md:int,
            dropout: float = 0.0,
        ):
        super(AttentiveFP_alt, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                    dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                            add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                    dropout=dropout, add_self_loops=False,
                                    negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
            # self.lin2 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels+num_md, out_channels)
            # self.lin3 = Linear(2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()
            # self.lin3.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, 
                    batch: Tensor,md: Tensor) -> Tensor:

        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

            # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

            # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = torch.cat([out, md], dim=1)
        out = (self.lin2(out))
            # # return out
            # out_add = torch.cat([out, i.to(device)], dim=1)
            # # return out_add
            # out_add = torch.sigmoid(self.lin3(out_add))
        return out


def make_graphlist(data, smile_col = 'smiles', y_col = 'y'):
    graph_list = []
    for index, row in tqdm(data.iterrows(), total = len(data)):
        g = from_smiles(row[smile_col])
        g.x = g.x.float()
        y = torch.tensor(row[y_col], dtype=torch.float).view(1, -1)
        g.y = y
        graph_list.append(g)
    return graph_list
    # graph_list = []
    # for index, row in data.iterrows():
    # g = from_smiles(row['smiles'])
    # g.x = g.x.float()
    # temp_i = Descriptors.MolLogP(Chem.MolFromSmiles(row['smiles']))
    # i = torch.tensor(temp_i, dtype=torch.float).view(1, -1)
    # y = torch.tensor(row['exp'], dtype=torch.float).view(1, -1)
    # g.y = y
    # g.i = i
    # graph_list.append(g)
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
def data_split(dataset, split_ratio = 0.8):
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Split the dataset into train and test subsets
    generator1 = torch.Generator().manual_seed(42)
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=generator1)
    # dataset_size = len(train_data)
    # train_size = int(split_ratio * dataset_size)
    # test_size = dataset_size - train_size
    # train_subset, val_subset = random_split(train_data, [train_size, test_size], generator=generator1)
    return(train_data, test_data)
    
def train_test_split(graph_list, split_ratio = 0.8):
    dataset_size = len(graph_list)
    train_size = int(split_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Split the dataset into train and test subsets
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator1)
    return(train_dataset, test_dataset)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # return train_loader, test_loade