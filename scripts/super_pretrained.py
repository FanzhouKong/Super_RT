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

from torch_geometric.loader import DataLoader
from dl_toolsets.super_model import Super, Super_2,train_super, test_super
from dl_toolsets.utilities import make_ecfp, make_aft_embedding, make_datalist, attentive_fp_featurizers



model_dir = ''
data_dir = '/Users/fanzhoukong/Documents/GitHub/Libgen_data/metlin_smrt'
data_subset = pd.read_csv(os.path.join(data_dir, 'SMRT_dataset_cleaned.csv'))
# data_all = pd.read_csv(os.path.join(data_dir, 'SMRT_parsed.csv'))
train_df, test_df = train_test_split(data_subset, test_size=0.1, random_state=42)

train_ecfp = make_ecfp(train_df['smiles'])
test_ecfp = make_ecfp(test_df['smiles'])
train_afp = attentive_fp_featurizers(train_df['smiles'])
test_afp = attentive_fp_featurizers(test_df['smiles'])
train_list = make_datalist(train_afp, train_ecfp,train_df['rt'].values)
test_list = make_datalist(test_afp, test_ecfp, test_df['rt'].values)
atom_feat = AttentiveFPAtomFeaturizer()
bond_feat = AttentiveFPBondFeaturizer()


print(f'there are {len(train_df)} samples in traning dataset')

train_loader = DataLoader(train_list, batch_size=64, shuffle=True, 
                        #   collate_fn=collate
                          )
test_loader = DataLoader(test_list, batch_size=64, shuffle=False,)

node_feat_size = atom_feat.feat_size()
edge_feat_size = bond_feat.feat_size()

model = Super_2(
    node_feat_size=node_feat_size,
    edge_feat_size=edge_feat_size,
    num_layers=2,   
    num_timesteps=2,
    graph_feat_size=512,
    feat_size = 512,
)
model = model.to('cpu')
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003126662000605776,)
max_epoch = 500
best_val_loss = float('inf')
saved = False
for i in range(max_epoch):
    train_super( i,model, train_loader, loss_fn,optimizer)
    val_rmse = test_super(model, test_loader)
    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        torch.save(model.state_dict(), 'models/best_super_2.pth')
        saved = True
        patience = 10
    else:
        patience -= 1
        if patience == 0:
            print('early stopping triggered')
            break
print('the training statistics are: ')
model.load_state_dict(torch.load('models/best_super_2.pth'))
print('this is the training statistics')
test_super(model, train_loader, verbose = True)
print('this is the testing statistics')
test_super(model, test_loader, verbose = True)