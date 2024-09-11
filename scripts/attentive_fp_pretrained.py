from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import Meter, EarlyStopping, SMILESToBigraph
import pandas as pd
from dgllife.data import MoleculeCSVDataset
import os
import dgl
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
atom_feat = AttentiveFPAtomFeaturizer()
bond_feat = AttentiveFPBondFeaturizer()

def load_dataset(df, atom_feat, bond_feat):
    smiles_to_g = SMILESToBigraph(add_self_loop=False, node_featurizer=atom_feat,
                                  edge_featurizer=bond_feat)
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=smiles_to_g,
                                 smiles_column='smiles',
                                 cache_file_path= 'models/graph.bin',
                                 task_names='rt',
                                 n_jobs=6)

    return dataset
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

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks
data_dir = '/Users/fanzhoukong/Documents/GitHub/Libgen_data/metlin_smrt'
data_subset = pd.read_csv(os.path.join(data_dir, 'SMRT_parsed.csv'))
# data_all = pd.read_csv(os.path.join(data_dir, 'SMRT_parsed.csv'))
train_df, test_df = train_test_split(data_subset, test_size=0.1, random_state=42)
print(f'there are {len(train_df)} samples in traning dataset')
train_data = load_dataset(train_df, atom_feat, bond_feat)
test_data = load_dataset(test_df, atom_feat, bond_feat)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True,
                              collate_fn=collate_molgraphs
                              )
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False,collate_fn=collate_molgraphs)

node_feat_size = atom_feat.feat_size()
edge_feat_size = bond_feat.feat_size()
def predict( model, bg):
    bg = bg.to('cpu')
    
    
    node_feats = bg.ndata.pop('h').to('cpu')
    edge_feats = bg.edata.pop('e').to('cpu')
        # return model(bg, node_feats, edge_feats)
    return model(bg, node_feats, edge_feats)
def train(model, data_loader,epoch, optimizer, criterion, scheduler = None):
    model.train()
    total_loss = 0
    total_items = 0
    for batch_id, batch_data in enumerate(data_loader):
        batch_n = len(batch_data[0])
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to('cpu'), masks.to('cpu')
        prediction = predict( model, bg)
        loss = (criterion(prediction, labels.view(-1, 1)) )
        total_loss += loss.item()*batch_n
        total_items += batch_n
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    median_loss = total_loss / total_items
    print(f'this is epoch {epoch},the loss is {np.sqrt(median_loss)}')
def test(model, data_loader, loss_criterion, verbose = False, return_value = True):
    model.eval()
    losses = []
    predictions = []
    refs = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            refs.extend(labels)

            if torch.cuda.is_available():
                bg.to(torch.device('cpu'))
                labels = labels.to('cpu')
                masks = masks.to('cpu')
            
            prediction = model(bg, bg.ndata['h'], bg.edata['e'])
            predictions.extend(prediction)
            loss = (loss_criterion(prediction, labels.view(-1, 1)).float())
            #loss = loss_criterion(prediction, labels)
            losses.append(loss.data.item())
    predictions= np.array([i.item() for i in predictions])
    refs = np.array([i.item() for i in refs])
    mse = loss_criterion(torch.tensor(predictions), torch.tensor(refs))
    mae = np.median(abs(predictions-refs))
    if verbose:
        print(f'the rmse is {np.sqrt(mse)}, the mae is {mae}')
    if return_value:
        return np.sqrt(mse)
    else:
        return ()
def get_mae(y_true, y_pred):
    mae = np.median(np.abs(y_pred - y_true))
    return mae
def get_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return rmse
model = AttentiveFPPredictor(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=2,
                                  num_timesteps=2,
                                  graph_feat_size=256,
                                  n_tasks=1,
                                  dropout=0.0)
model = model.to('cpu')
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003126662000605776,)
max_epoch = 500
best_val_loss = float('inf')
saved = False
for i in range(0, max_epoch):
    train(model, train_loader, i, optimizer, loss_fn)
    val_rmse = test(model, test_loader, loss_fn).item()
    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        torch.save(model.state_dict(), 'models/best_my_trained_attfp.pth')
        saved = True
        patience = 10
    else:
        patience -= 1
        if patience == 0:
            print('early stopping triggered')
            break
print('the training statistics are: ')
test(model, train_loader, loss_fn, verbose = True, return_value=False)
print('the testing statistics are: ')
test(model, test_loader, loss_fn, verbose = True,return_value=False)
if saved == False:
    torch.save(model.state_dict(), 'models/last_my_trained_attfp.pth')