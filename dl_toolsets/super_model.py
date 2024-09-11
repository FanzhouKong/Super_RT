from dgllife.model.readout import AttentiveFPReadout
from dgllife.model.gnn import AttentiveFPGNN
import torch.nn as nn
import torch
from dl_toolsets.ResGat import OneDimConvBlock
from dl_toolsets.utilities import predict_super, predict
import numpy as np



def train(epoch, model, data_loader, loss_criterion, optimizer, device = 'cpu', scheduler = None):
    model.train()

    total_loss = 0
    total_items = 0
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        prediction = predict(model, batch_data)
        loss = loss_criterion(prediction, batch_data.y)
        total_loss += loss.detach().item()*len(batch_data.y)
        total_items += len(batch_data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    epoch_rmse = np.sqrt(total_loss/total_items)
    print(f'Epoch {epoch}, RMSE: {epoch_rmse}')
    # return total_loss / (batch_id + 1)
def train_super(epoch, model, data_loader, loss_criterion, optimizer, device = 'cpu', super = True):
    model.train()

    total_loss = 0
    total_items = 0
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        if super == True:
            prediction = predict_super( model, batch_data)
        # else:
        #     prediction = predict(model, batch_data.ecfp)
        loss = loss_criterion(prediction, batch_data.y)
        total_loss += loss.detach().item()*len(batch_data.y)
        total_items += len(batch_data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_rmse = np.sqrt(total_loss/total_items)
    print(f'Epoch {epoch}, RMSE: {epoch_rmse}')
    # return total_loss / (batch_id + 1)

def test_super(model, data_loader, verbose = False, return_values = False, return_rmse = True, super = True):
    model.eval()
    with torch.no_grad():

        preds = []
        refs = []
        for batch_data in data_loader:
            if super == True:
                prediction = predict_super( model, batch_data)
            # else:
            #     prediction = predict(model, batch_data.ecfp)

            preds.extend(prediction)
            refs.extend(batch_data.y)
    refs = np.array([x.item() for x in refs])
    preds = np.array([x.item() for x in preds])
    loss_fn = nn.MSELoss()
    mse = loss_fn(torch.tensor(preds), torch.tensor(refs))
    rmse = np.sqrt(mse)
    mae = np.median(abs(preds-refs))
    if verbose:
        print(f'the testing rmse is {rmse}, the mae is {mae}')
    if return_values:
        return refs, preds
    if return_rmse:
        return rmse
def test(model, data_loader, verbose = False, return_values = False, return_rmse = True):
    model.eval()
    with torch.no_grad():

        preds = []
        refs = []
        for batch_data in data_loader:
            prediction = predict(model, batch_data)
            preds.extend(prediction)
            refs.extend(batch_data.y)
    
    refs = np.array([x.item() for x in refs])
    
    preds = np.array([x.item() for x in preds])
    
    loss_fn = nn.MSELoss()
    
    mse = loss_fn(torch.tensor(preds), torch.tensor(refs))
    
    rmse = np.sqrt(mse)
    mae = np.median(abs(preds-refs))
    if verbose:
        print(f'the testing rmse is {rmse}, the mae is {mae}')
    if return_values:
        return refs, preds
    if return_rmse:
        return rmse
class Super_2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, num_timesteps, graph_feat_size, feat_size, dropout=0):
        super(Super_2, self).__init__()
        self.feat_size = feat_size
        self.drop_ratio = dropout
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.feat_lin = nn.Linear(graph_feat_size, graph_feat_size)
        self.conv1d1 = OneDimConvBlock()
        self.conv1d2 = OneDimConvBlock()
        self.conv1d3 = OneDimConvBlock()
        self.conv1d4 = OneDimConvBlock()
        self.conv1d5 = OneDimConvBlock()
        self.conv1d6 = OneDimConvBlock()
        self.conv1d7 = OneDimConvBlock()
        self.conv1d8 = OneDimConvBlock()
        self.conv1d9 = OneDimConvBlock()
        self.conv1d10 = OneDimConvBlock()
        self.conv1d11 = OneDimConvBlock()
        self.conv1d12 = OneDimConvBlock()

        self.preconcat1 = nn.Linear(2048, 1024)
        self.preconcat2 = nn.Linear(1024, 512)#this is for descriptors

        self.afterconcat1 = nn.Linear(2 * self.feat_size, self.feat_size)
        self.after_cat_drop = nn.Dropout(self.drop_ratio)
        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_size, self.feat_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_size // 8, self.feat_size // 64),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_size // 64, 1),
        )
    def forward(self,g, node_feats, edge_feats, 
                fringerprint,
                get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats, get_node_weight)
        # return g_feats
        g_feats = self.feat_lin(g_feats)

        fringerprint = self.conv1d1(fringerprint)
        fringerprint = self.conv1d2(fringerprint)
        fringerprint = self.conv1d3(fringerprint)
        fringerprint = self.conv1d4(fringerprint)
        fringerprint = self.conv1d5(fringerprint)
        fringerprint = self.conv1d6(fringerprint)
        fringerprint = self.conv1d7(fringerprint)
        fringerprint = self.conv1d8(fringerprint)
        fringerprint = self.conv1d9(fringerprint)
        fringerprint = self.conv1d10(fringerprint)
        fringerprint = self.conv1d11(fringerprint)
        fringerprint = self.conv1d12(fringerprint)

        fringerprint = self.preconcat1(fringerprint)
        fringerprint = self.preconcat2(fringerprint)
        concat = torch.concat([g_feats, fringerprint], dim=-1)
        concat = self.afterconcat1(concat)
        
        concat = self.after_cat_drop(concat)
        out = self.out_lin(concat)
        return out




class TransforerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TransforerBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1d1 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d2 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d3 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d4 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d5 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d6 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d7 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d8 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d9 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d10 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d11 = OneDimConvBlock(self.in_channel,self.out_channel)
        self.conv1d12 = OneDimConvBlock(self.in_channel,self.out_channel)
    def forward(self, fringerprint):
        fringerprint = self.conv1d1(fringerprint)
        fringerprint = self.conv1d2(fringerprint)
        fringerprint = self.conv1d3(fringerprint)
        fringerprint = self.conv1d4(fringerprint)
        fringerprint = self.conv1d5(fringerprint)
        fringerprint = self.conv1d6(fringerprint)
        fringerprint = self.conv1d7(fringerprint)
        fringerprint = self.conv1d8(fringerprint)
        fringerprint = self.conv1d9(fringerprint)
        fringerprint = self.conv1d10(fringerprint)
        fringerprint = self.conv1d11(fringerprint)
        fringerprint = self.conv1d12(fringerprint)
        # fringerprint = self.preconcat1(fringerprint)
        # fringerprint = self.preconcat2(fringerprint)
        return fringerprint

class Super(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, num_timesteps, graph_feat_size, feat_size, dropout=0):
        super(Super, self).__init__()
        self.feat_size = feat_size
        self.drop_ratio = dropout
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.feat_lin = nn.Linear(graph_feat_size, self.feat_size)
        self.conv1d = TransforerBlock(in_channel=2048, out_channel=2048)

        self.preconcat1 = nn.Linear(2048, 1024)
        self.preconcat2 = nn.Linear(1024, self.feat_size)#this is for descriptors

        self.afterconcat1 = nn.Linear(2 * self.feat_size, self.feat_size)
        self.after_cat_drop = nn.Dropout(self.drop_ratio)
        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_size, self.feat_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_size // 8, self.feat_size // 64),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_size // 64, 1),
        )
        # self.out_lin2 = nn.Sequential(
        #     nn.Linear(202, self.feat_size // 8),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.feat_size // 8, self.feat_size // 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.feat_size // 64, 1),
        # )
        # self.final_predict = nn.Sequential(
        #     nn.Linear(2, 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2, 1)
        # )
    def forward(self,g, node_feats, edge_feats, 
                fingerprint,
                get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats, get_node_weight)

        g_feats = self.feat_lin(g_feats)

        fingerprint = self.conv1d(fingerprint)
        fingerprint = self.preconcat1(fingerprint)
        fingerprint = self.preconcat2(fingerprint)
        concat = torch.concat([g_feats, fingerprint], dim=-1)
        concat = self.afterconcat1(concat)
        
        concat = self.after_cat_drop(concat)
        out = self.out_lin(concat)
        return out
        # return g_feats
        # fringerprint = data.fingerprint.reshape(-1, 2048)









        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr
        # batch = data.batch
        # fringerprint = data.fingerprint.reshape(-1, 2048)

        # h = self.in_linear(x)

        # h = F.relu(self.conv1(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv2(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv3(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv4(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv5(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv6(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv7(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv8(h, edge_index, edge_attr), inplace=True)
        # h = F.relu(self.conv9(h, edge_index, edge_attr), inplace=True)

        # fringerprint = self.conv1d1(fringerprint)
        # fringerprint = self.conv1d2(fringerprint)
        # fringerprint = self.conv1d3(fringerprint)
        # fringerprint = self.conv1d4(fringerprint)
        # fringerprint = self.conv1d5(fringerprint)
        # fringerprint = self.conv1d6(fringerprint)
        # fringerprint = self.conv1d7(fringerprint)
        # fringerprint = self.conv1d8(fringerprint)
        # fringerprint = self.conv1d9(fringerprint)
        # fringerprint = self.conv1d10(fringerprint)
        # fringerprint = self.conv1d11(fringerprint)
        # fringerprint = self.conv1d12(fringerprint)
        # fringerprint = self.preconcat1(fringerprint)
        # fringerprint = self.preconcat2(fringerprint)

        # h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        # h = self.pool(h, batch)
        # h = self.feat_lin(h)

        # concat = torch.concat([h, fringerprint], dim=-1)
        # concat = self.afterconcat1(concat)
        # concat = self.after_cat_drop(concat)

        # out = self.out_lin(concat)












# class RT_AFP(nn.Module):
#     def __init__(self,
#                 node_feat_size,
#                 edge_feat_size,
#                 num_layers=2,
#                 num_timesteps=2,
#                 graph_feat_size=200,
#                 n_tasks=1,
#                 dropout=0.):
#         super(RT_AFP, self).__init__()
#         self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
#                                   edge_feat_size=edge_feat_size,
#                                   num_layers=num_layers,
#                                   graph_feat_size=graph_feat_size,
#                                   dropout=dropout)
#         self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
#                                           num_timesteps=num_timesteps,
#                                           dropout=dropout)
#         self.predict = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(graph_feat_size, n_tasks)
#         )
#     def forward(self, g, node_feats, edge_feats, get_node_weight=False):
#         node_feats = self.gnn(g, node_feats, edge_feats)
#         if get_node_weight:
#             g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
#             return self.predict(g_feats), node_weights
#         else:
#             g_feats = self.readout(g, node_feats, get_node_weight)
#             return self.predict(g_feats)





# class OG_AFP(nn.Module):
#     def __init__(self,
#                 node_feat_size,
#                 edge_feat_size,
#                 num_layers=2,
#                 num_timesteps=2,
#                 graph_feat_size=200,
#                 n_tasks=1,
#                 dropout=0.):
#         super(RT_AFP, self).__init__()
#         self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
#                                   edge_feat_size=edge_feat_size,
#                                   num_layers=num_layers,
#                                   graph_feat_size=graph_feat_size,
#                                   dropout=dropout)
#         self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
#                                           num_timesteps=num_timesteps,
#                                           dropout=dropout)
#         self.predict = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(graph_feat_size, n_tasks)
#         )
#     def forward(self, g, node_feats, edge_feats, get_node_weight=False):
#         node_feats = self.gnn(g, node_feats, edge_feats)
#         if get_node_weight:
#             g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
#             return self.predict(g_feats), node_weights
#         else:
#             g_feats = self.readout(g, node_feats, get_node_weight)
#             return self.predict(g_feats)