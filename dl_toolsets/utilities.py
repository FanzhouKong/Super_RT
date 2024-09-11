from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import Meter, EarlyStopping, SMILESToBigraph
import pandas as pd
from dgllife.data import MoleculeCSVDataset
import os
from torch_geometric.data import Dataset, Data
import dgl
from dgl.nn.pytorch.glob import AvgPooling
import numpy as np
from rdkit.Chem import Descriptors
from dgllife.model import load_pretrained
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
def make_datalist(aft_graphs, ecfp_fgps, 
                    # gin_embds,
               #    rdkit_dcps, 
                  rt_list):
     data_list = []
     for i in range(len(aft_graphs)):
          bg = aft_graphs[i]
          data_row = Data(
               bg = bg,
               # gin = torch.tensor(gin_embds[i], dtype=torch.float32).view(1, -1),
               ecfp = torch.tensor(ecfp_fgps[i], dtype=torch.float32).view(1, -1),
               # dcps = torch.tensor(rdkit_dcps[i], dtype=torch.float32).view(1, -1),
               
                         #  node_feat = bg.ndata.pop('h'),
                         #      edge_feat = test[i].edata.pop('e'),
          )
          y = torch.tensor(rt_list[i], dtype=torch.float).view(1, -1)
          data_row.y = y
          # data_row.gin = gin
          data_list.append(data_row)
     return data_list
def predict_super( model, data):

    bg = flatten_bg(data.bg)
    device = 'cpu'
    bg = bg.to(device)
    node_feats = bg.ndata.pop('h').to(device)
    edge_feats = bg.edata.pop('e').to(device)
    return model(bg, node_feats, edge_feats, data.ecfp)
def predict( model, data):
    bg = flatten_bg(data.bg)
    device = 'cpu'
    bg = bg.to(device)
    node_feats = bg.ndata.pop('h').to(device)
    edge_feats = bg.edata.pop('e').to(device)
    
    return model(bg, node_feats, edge_feats)
def flatten_bg(dbg):
    bg  = dgl.batch(dbg)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return(bg)



def make_rdkit_descriptors(smiles, use_BCUT = False):
    import deepchem as dc
    all_descriptors = {name: func for name, func in Descriptors.descList}
    all_names = list((all_descriptors.keys()))
    if use_BCUT == False:
        all_names = [x for x in all_names if x.startswith('BCUT2D_')== False]
    f2 = dc.feat.RDKitDescriptors(use_bcut2d= use_BCUT)
    feats2 = f2.featurize(smiles)
    return(feats2)



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
# data_dir = '/Users/fanzhoukong/Documents/GitHub/Libgen_data/metlin_smrt'
# data_subset = pd.read_csv(os.path.join(data_dir, 'SMRT_subset.csv'))
# data_all = pd.read_csv(os.path.join(data_dir, 'SMRT_parsed.csv'))
# train_df, test_df = train_test_split(data_subset, test_size=0.1, random_state=42)
# train_data = load_dataset(train_df, atom_feat, bond_feat)
# test_data = load_dataset(test_df, atom_feat, bond_feat)
# train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True,
#                               collate_fn=collate_molgraphs
#                               )
# test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False,collate_fn=collate_molgraphs)

# node_feat_size = atom_feat.feat_size()
# edge_feat_size = bond_feat.feat_size()

# def train(model, data_loader,epoch, optimizer, criterion, scheduler = None):
#     model.train()
#     total_loss = 0
#     total_items = 0
#     for batch_id, batch_data in enumerate(data_loader):
#         batch_n = len(batch_data[0])
#         smiles, bg, labels, masks = batch_data
#         if len(smiles) == 1:
#             # Avoid potential issues with batch normalization
#             continue

#         labels, masks = labels.to('cpu'), masks.to('cpu')
#         prediction = predict( model, bg)
#         loss = (criterion(prediction, labels.view(-1, 1)) )
#         total_loss += loss.item()*batch_n
#         total_items += batch_n
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if scheduler is not None:
#             scheduler.step()
#     median_loss = total_loss / total_items
#     print(f'this is epoch {epoch},the loss is {np.sqrt(median_loss)}')
# def test(model, data_loader, loss_criterion, verbose = False, return_rmse = True, return_values = False):
#     model.eval()
#     losses = []
#     predictions = []
#     refs = []
#     with torch.no_grad():
#         for batch_id, batch_data in enumerate(data_loader):
#             smiles, bg, labels, masks = batch_data
#             refs.extend(labels)

#             if torch.cuda.is_available():
#                 bg.to(torch.device('cpu'))
#                 labels = labels.to('cpu')
#                 masks = masks.to('cpu')
            
#             prediction = model(bg, bg.ndata['h'], bg.edata['e'])
#             predictions.extend(prediction)
#             loss = (loss_criterion(prediction, labels.view(-1, 1)).float())
#             #loss = loss_criterion(prediction, labels)
#             losses.append(loss.data.item())
#     predictions= np.array([i.item() for i in predictions])
#     refs = np.array([i.item() for i in refs])
#     mse = loss_criterion(torch.tensor(predictions), torch.tensor(refs))
#     mae = np.median(abs(predictions-refs))
#     if verbose:
#         print(f'the rmse is {np.sqrt(mse)}, the mae is {mae}')
#     if return_values == True:
#         return predictions, refs
#     elif return_rmse:
#         return np.sqrt(mse)
#     else:
#         return ()
def get_mae(y_true, y_pred):
    mae = np.median(np.abs(y_pred - y_true))
    return mae
def get_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return rmse

import torch.nn.functional as F
def make_ecfp(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mols =[Chem.MolFromSmiles(smi) for smi in smiles]
    fp = []
    for m in tqdm(mols):
        fp.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048, useFeatures = False, useChirality = False))

    return np.array(fp, dtype = np.float32)
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state):
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        
        # Compute attention scores
        attn_scores = self.attention_weights(last_hidden_state)  # (batch_size, sequence_length, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch_size, sequence_length)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, sequence_length)
        
        # Multiply hidden states by the attention weights
        weighted_hidden_state = last_hidden_state * attn_weights.unsqueeze(-1)  # (batch_size, sequence_length, hidden_size)
        
        # Sum the weighted hidden states to get the final representation
        pooled_output = weighted_hidden_state.sum(dim=1)  # (batch_size, hidden_size)
        
        return pooled_output
from transformers import BertTokenizerFast, BertModel
def make_clf_embeddings(smiles, attention_pooling=False, default_pooler = False):
    checkpoint = 'unikei/bert-base-smiles'
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    model = BertModel.from_pretrained(checkpoint)
    
    with torch.no_grad():
        embeddings_all = torch.empty(0)
        for s in tqdm(smiles):
            tokens = tokenizer(s, return_tensors='pt')
            predictions = model(**tokens)
            if attention_pooling == True:
                attention_pooling = AttentionPooling(hidden_size=768)
                embeddings = attention_pooling(predictions.last_hidden_state)
            elif default_pooler == True:
                embeddings = predictions.pooler_output
            else:
                embeddings = predictions[0][:,0]
            embeddings_all = torch.cat((embeddings_all, embeddings[0].unsqueeze(0)), dim=0)
        
    return embeddings_all.detach().numpy()
def make_dataset(embeddings, labels):
    labels = torch.from_numpy(labels.astype(np.float32))
    labels  = labels.view(labels.shape[0],1)
    embeddings = torch.from_numpy(embeddings.astype(np.float32))
    dataset = TensorDataset( embeddings, labels)
    return dataset
def make_data(embeddings, labels):
    data_list = []
    for i in range(len(embeddings)):
        data = Data(
            emb = torch.tensor(embeddings[i], dtype = torch.float32),
            y = torch.tensor(labels[i], dtype = torch.float32) 
        )
        data_list.append(data)
    return data_list
def gin_featurizers(smiles):
    """Construct graphs from SMILES and featurize them

    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation

    Returns
    -------
    list of DGLGraph
        List of graphs constructed and featurized
    list of bool
        Indicators for whether the SMILES string can be
        parsed by RDKit
    """
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)

    return graphs, success
def attentive_fp_featurizers(smiles):
    graphs = []
    success = []
    for smi in tqdm(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=False,
                               node_featurizer=AttentiveFPAtomFeaturizer(),
                               edge_featurizer=AttentiveFPBondFeaturizer(),
                            #    canonical_atom_order=False
                               )
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)
    return graphs
def collate(graphs):
    return dgl.batch(graphs)
def make_gin_embedding(smiles, model_name='gin_supervised_contextpred'):
    gin_feats = gin_featurizers(smiles)
    dataset = [gin_feats[0][x] for x in range(len(gin_feats[1])) if gin_feats[1][x] == True]# if there is fail?
    args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model': model_name,
        'batch_size': 128
    }
    data_loader = DataLoader(dataset, batch_size=args['batch_size'],
                             collate_fn=collate, shuffle=False)
    
    model = load_pretrained(args['model']).to(args['device'])
    model.eval()
    readout = AvgPooling()
    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        print('Processing batch {:d}/{:d}'.format(batch_id + 1, len(data_loader)))
        bg = bg.to(args['device'])
        nfeats = [bg.ndata.pop('atomic_number').to(args['device']),
                    bg.ndata.pop('chirality_type').to(args['device'])]
        efeats = [bg.edata.pop('bond_type').to(args['device']),
                    bg.edata.pop('bond_direction_type').to(args['device'])]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        mol_emb.append(readout(bg, node_repr))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    return mol_emb
class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x
def load_apf_emb_model(path = 'models/best_my_trained_attfp.pth'):
    model_featuizer = AttentiveFPPredictor(node_feat_size=AttentiveFPAtomFeaturizer().feat_size(),
                                  edge_feat_size=AttentiveFPBondFeaturizer().feat_size(),

                                  num_layers=2,
                                  num_timesteps=2,
                                  graph_feat_size=256,
                                  n_tasks=1,
                                  dropout=0.0)
    model_featuizer.load_state_dict(torch.load(path))
    
    model_featuizer.predict = Identity()
    return model_featuizer
def make_aft_embedding(smiles):
    model = load_apf_emb_model()
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=False,
                               node_featurizer=AttentiveFPAtomFeaturizer(),
                               edge_featurizer=AttentiveFPBondFeaturizer(),
                            #    canonical_atom_order=False
                               )
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)
    model.eval()
    afp_embedding = []
    with torch.no_grad():
        for bg in graphs:
            bg = bg.to('cpu')
            node_feats = bg.ndata.pop('h').to('cpu')
            edge_feats = bg.edata.pop('e').to('cpu')
            afp_embedding.append(model(bg, node_feats, edge_feats))
            # break
    afp_embedding = torch.cat(afp_embedding, dim=0).detach().cpu().numpy()
    return afp_embedding