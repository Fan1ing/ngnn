import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from torch_geometric.nn import NNConv, Set2Set, GATv2Conv, GCNConv, GINEConv,SetTransformerAggregation
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
csv_path = 'C:/Users/Arthas/Desktop/fanjinming1.csv'
df = pd.read_csv(csv_path)
y1 = df['activity']


smiles = df['smile']
ys = y1
print(ys)

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()

    def isInRing(self, atom):
        return atom.IsInRing()

    def isaromatic(self, atom):
        return atom.GetIsAromatic()

    def formal_charge(self, atom):
        return atom.GetFormalCharge()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()
    def conjugated(self, bond):
        return bond.GetIsConjugated()


    def aromatic(self, bond):
        return bond.GetIsAromatic()

    def ring(self, bond):
        return bond.IsInRing()

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Cl", "F","Ge", "H", "I", "N", "Na", "O", "P", "S","Se","Si","Te"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "formal_charge": {-1, -2, 1, 2, 0},
        "hybridization": {"s", "sp", "sp2", "sp3"},
        "isInRing": {True, False},
        "isaromatic": {True, False},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
        "aromatic": {True, False},
        "ring": {True, False},
    }
)


# mol = Chem.MolFromSmiles('CO')
# mol = Chem.AddHs(mol)
# # for atom in mol.GetAtoms():
# #     # print(atom.GetSymbol())
# #     # print(atom_featurizer.encode(atom))
# for bond in mol.GetBonds():
#     print([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     print(bond_featurizer.encode(bond))


class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, transform = None):
        super(MoleculesDataset,self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        datas = []
        for smile, y in zip(smiles,ys):
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)

            embeddings = []
            for atom in mol.GetAtoms():
                embeddings.append(atom_featurizer.encode(atom))
            embeddings = torch.tensor(embeddings,dtype=torch.float32)

            #增加Nr
            rows, cols = embeddings.shape
            zeros_tensor = torch.zeros(rows, 7)
            embeddings = torch.cat((embeddings,zeros_tensor),dim = 1)
            for i in range(rows):
        #B
                if embeddings[i,0] == 1:
                    embeddings[i,-1] = 2.04 #电负性
                    embeddings[i,-2] = 82  #共价半径
                    embeddings[i,-3] = 5   #原子序数
                    embeddings[i,-4] = 10.82 #原子质量
                    embeddings[i,-5] = 8.298 #第一电离能
                    embeddings[i,-6] = 0.277 #电子亲合能
                    embeddings[i,-7] = 0

                #Br
                elif embeddings[i,1] == 1:
                    embeddings[i,-1] = 2.96
                    embeddings[i,-2] = 114
                    embeddings[i,-3] = 35
                    embeddings[i,-4] = 79.904
                    embeddings[i,-5] = 11.814
                    embeddings[i,-6] = 3.364
                    embeddings[i,-7] = 1

                #C
                elif embeddings[i,2] == 1:
                    embeddings[i,-1] = 2.55
                    embeddings[i,-2] = 77
                    embeddings[i,-3] = 6
                    embeddings[i,-4] = 12.011
                    embeddings[i,-5] = 11.261
                    embeddings[i,-6] = 1.595
                    embeddings[i,-7] = 0

                #Cl
                elif embeddings[i,3] == 1:
                    embeddings[i,-1] = 3.16
                    embeddings[i,-2] = 99
                    embeddings[i,-3] = 17
                    embeddings[i,-4] = 35.45
                    embeddings[i,-5] = 12.968
                    embeddings[i,-6] = 3.62
                    embeddings[i,-7] = 1
                #F
                elif embeddings[i,4] == 1:
                    embeddings[i,-1] = 3.98
                    embeddings[i,-2] = 71
                    embeddings[i,-3] = 9
                    embeddings[i,-4] = 18.998
                    embeddings[i,-5] = 17.422
                    embeddings[i,-6] = 3.40
                    embeddings[i,-7] = 1

            #Ge
                elif embeddings[i,5] == 1:
                    embeddings[i,-1] = 2.01
                    embeddings[i,-2] = 122
                    embeddings[i,-3] = 32
                    embeddings[i,-4] = 72.63
                    embeddings[i,-5] = 7.90
                    embeddings[i,-6] = 1.23
                    embeddings[i,-7] = 0


                #H
                elif embeddings[i,6] == 1:
                    embeddings[i,-1] = 2.20
                    embeddings[i,-2] = 37
                    embeddings[i,-3] = 1
                    embeddings[i,-4] = 1.008
                    embeddings[i,-5] = 13.598
                    embeddings[i,-6] = 0.755
                    embeddings[i,-7] = 0


                #I
                elif embeddings[i,7] == 1:
                    embeddings[i,-1] = 2.66
                    embeddings[i,-2] = 133
                    embeddings[i,-3] = 53
                    embeddings[i,-4] = 126.9
                    embeddings[i,-5] = 10.451
                    embeddings[i,-6] = 3.060
                    embeddings[i,-7] = 1

                 #N
                elif embeddings[i,8] == 1:
                    embeddings[i,-1] = 3.04
                    embeddings[i,-2] = 75
                    embeddings[i,-3] = 7
                    embeddings[i,-4] = 14.007
                    embeddings[i,-5] = 14.534
                    embeddings[i,-6] = 0.07
                    embeddings[i,-7] = 0

                 #Na
                elif embeddings[i,9] == 1:
                    embeddings[i,-1] = 0.93
                    embeddings[i,-2] = 154
                    embeddings[i,-3] = 11
                    embeddings[i,-4] = 22.99
                    embeddings[i,-5] = 5.139
                    embeddings[i,-6] = 0.547
                    embeddings[i,-7] = 0

                 #O
                elif embeddings[i,10] == 1:
                    embeddings[i,-1] = 3.44
                    embeddings[i,-2] = 73
                    embeddings[i,-3] = 8
                    embeddings[i,-4] = 15.999
                    embeddings[i,-5] = 13.618
                    embeddings[i,-6] = 1.46
                    embeddings[i,-7] = 0
                 #P
                elif embeddings[i,11] == 1:
                    embeddings[i,-1] = 2.19 #电负性
                    embeddings[i,-2] = 106  #共价半径
                    embeddings[i,-3] = 15   #原子序数
                    embeddings[i,-4] = 30.974 #原子质量
                    embeddings[i,-5] = 10.487 #第一电离能
                    embeddings[i,-6] = 0.75 #电子亲合能
                    embeddings[i,-7] = 0


                #S
                elif embeddings[i,12] == 1:
                    embeddings[i,-1] = 2.58 #电负性
                    embeddings[i,-2] = 102  #共价半径
                    embeddings[i,-3] = 16   #原子序数
                    embeddings[i,-4] = 32.06 #原子质量
                    embeddings[i,-5] = 10.36 #第一电离能
                    embeddings[i,-6] = 2.07 #电子亲合能
                    embeddings[i,-7] = 0

                #Se
                elif embeddings[i,13] == 1:
                    embeddings[i,-1] = 2.55 #电负性
                    embeddings[i,-2] = 116  #共价半径
                    embeddings[i,-3] = 34   #原子序数
                    embeddings[i,-4] = 78.971 #原子质量
                    embeddings[i,-5] = 9.753 #第一电离能
                    embeddings[i,-6] = 2.02 #电子亲合能
                    embeddings[i,-7] = 0
                #Si
                elif embeddings[i,14] == 1:
                    embeddings[i,-1] = 1.90 #电负性
                    embeddings[i,-2] = 111  #共价半径
                    embeddings[i,-3] = 14   #原子序数
                    embeddings[i,-4] = 28.085 #原子质量
                    embeddings[i,-5] = 8.151 #第一电离能
                    embeddings[i,-6] = 1.385 #电子亲合能
                    embeddings[i,-7] = 0

                #Te
                elif embeddings[i,15] == 1:
                    embeddings[i,-1] = 2.1 #电负性
                    embeddings[i,-2] = 135  #共价半径
                    embeddings[i,-3] = 52   #原子序数
                    embeddings[i,-4] = 127.6 #原子质量
                    embeddings[i,-5] = 9.010 #第一电离能
                    embeddings[i,-6] = 1.971 #电子亲合能
                    embeddings[i,-7] = 0

            embeddings = embeddings[:, 16:]
            embeddings = torch.tensor(embeddings,dtype=torch.float32)



            edges = []
            edge_attr = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

                edge_attr.append(bond_featurizer.encode(bond))
                edge_attr.append(bond_featurizer.encode(bond))

            edges = torch.tensor(edges).T
            edge_attr = torch.tensor(edge_attr,dtype=torch.float32)

            y = torch.tensor(y,dtype=torch.float32)


            data = Data(x=embeddings, y=y, edge_index=edges, edge_attr=edge_attr)
            datas.append(data)

        # self.data, self.slices = self.collate(datas)
        torch.save(self.collate(datas), self.processed_paths[0])

max_nodes = 128
dataset = MoleculesDataset(root= "data")
#


# Split datasets.
train_size = int(0.8 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,valid_size, test_size]
                                                                           ,generator=torch.Generator().manual_seed(991))
print(train_size)
print(valid_size)
print(test_size)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# for i in train_loader:
#     print(i.edge_attr.shape)

class NGNN(nn.Module):
    def __init__(self,node_feature_dim, edge_feature_dim, edge_hidden_dim):
        super(NGNN,self).__init__()
        #第一层
        self.a1 = nn.Linear(6, 6)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.m1 = nn.Linear(26, 26)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        edge_network1 = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(edge_hidden_dim, 6 * 6)
        )
        self.nnconv1 = NNConv(6, 6, edge_network1, aggr="mean")

        edge_network2 = nn.Sequential(
            nn.Linear(6, 6),
        )
        self.GIN2 = GINEConv(edge_network2, edge_dim=10)

        # 第二层
        edge_networkx = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim, 6 * 12)
        )
        self.nnconvx = NNConv(6, 12, edge_networkx, aggr="mean")
        self.relu = nn.ReLU()

        edge_network3 = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim,32 * 64)
        )
        self.nnconv3 = NNConv(32, 64, edge_network3, aggr="mean")
        edge_network4 = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim,64 * 128)
        )
        self.nnconv4 = NNConv(64, 128, edge_network4, aggr="mean")
        edge_network5 = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim,128 * 256)
        )
        self.nnconv5 = NNConv(128, 256, edge_network5, aggr="mean")
        self.set2set = Set2Set(256, processing_steps=2)
        self.fc2 = nn.Linear(2*256, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=0.1)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, edge_attr,batch = data.x, data.edge_index, data.edge_attr,data.batch
        x1 = x[:, -6:]
        x1 = self.a1(x1)
        #x1 = self.nnconv1(x1, edge_index, edge_attr)
        x1 = self.relu(x1)
        #x1 = self.sigmoid(x1)
        x2 = x[:, :-6]
        x2 = self.m1(x2)
        #x2 = self.m1(x2)
        #x2 = self.nnconv2(x2, edge_index, edge_attr)
        x2 = self.relu(x2)
        #x2 = self.sigmoid(x2)

        c = torch.cat((x1, x2), dim = 1)
        x = c
        x = self.nnconv3(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.nnconv4(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.nnconv5(x, edge_index, edge_attr)
        x = self.set2set(x, batch)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


batch_size = 64


node_feature_dim, edge_feature_dim, edge_hidden_dim = 32 , 10 , 32

# conv = NNConvNet(node_feature_dim, edge_feature_dim, edge_hidden_dim)
#
# for i in train_loader :
#     print(conv(i))

num = 1000
lr = 0.001#0.001已测

model = NGNN(node_feature_dim, edge_feature_dim, edge_hidden_dim)
model = model.cuda()
#print(model)

optimizer = torch.optim.Adam(model.parameters(),lr)
criterion = torch.nn.MSELoss()
criterion = criterion.cuda()
#loss
loss_1 = []
train_mae = []
train_r2 = []
test_mae = []
test_r2 = []
for e in tqdm(range(num)):
    print('Epoch {}/{}'.format(e + 1, num))
    print('-------------')
    model.train()
    epoch_loss = []

    train_total = 0
    train_correct = 0

    train_preds = []
    train_trues = []

    #train
    for data in train_loader:
        y = data.y
        y = y.cuda()
        data = data.cuda()
        optimizer.zero_grad()
        out = model(data)
        # print(y.dtype, out.dtype)
        out = out.reshape(-1)
        y = y.reshape(-1)
        loss = criterion(out.float(), y.float())
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.data)
        train_preds.extend(out.detach().cpu().numpy())
        train_trues.extend(y.detach().cpu().numpy())
    mae = mean_absolute_error(train_trues,train_preds)
    r2 = r2_score( train_trues,train_preds)
    rmse = sqrt(mean_squared_error(train_trues,train_preds))
    mse = mean_squared_error(train_trues,train_preds)
    train_mae.append(mae)
    train_r2.append(r2)
    print("MAE OF TRAIN: {} ,RMSE : {} ,MSE : {} ,R2: {} ".format(mae,rmse,mse,r2))
        #评估指标
    # scheduler.step(epoch_loss)

    # early_stopping(epoch_loss, model)
    # # 若满足 early stopping 要求
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     # 结束模型训练
    #     break

    #f1_score

    #----------------------------------------------------------valid----------------------------------------------------
    val_total = 0
    val_correct = 0

    val_preds = []
    val_trues = []
    with torch.no_grad():
        model.eval()
        for data in val_loader:
            y = data.y
            y = y.cuda()
            data = data.cuda()
            outputs = model(data)
            outputs = outputs.reshape(-1)
            y = y.reshape(-1)
            val_preds.extend(outputs.detach().cpu().numpy())
            val_trues.extend(y.detach().cpu().numpy())
        mae = mean_absolute_error(val_trues,val_preds)
        r2 = r2_score(val_trues,val_preds)
        rmse = sqrt(mean_squared_error(val_trues,val_preds))
        mse = mean_squared_error(val_trues,val_preds)
        test_mae.append(mae)
        test_r2.append(r2)

        print("MAE OF test: {} ,RMSE : {} ,MSE : {} ,R2: {} ".format(mae,rmse,mse,r2))

pd.DataFrame(train_preds).to_csv('sample1.csv')
pd.DataFrame(train_trues).to_csv('sample2.csv')
pd.DataFrame(val_preds).to_csv('sample3.csv')
pd.DataFrame(val_trues).to_csv('sample4.csv')

