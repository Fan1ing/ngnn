import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from torch_geometric.nn import NNConv, Set2Set
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error,mean_absolute_percentage_error
from math import sqrt
csv_path = 'C:/Users/Arthas/Desktop/fanjinming.csv'
df = pd.read_csv(csv_path)
fan = pd.read_csv('C:/Users/Arthas/Desktop/data_E.csv',header=None)
fan= fan.to_numpy()
fan1 = pd.read_csv('C:/Users/Arthas/Desktop/data_G.csv',header=None)
fan1= fan1.to_numpy()
Toluene1 = fan1[0, :]

THF1= fan1[1, :]

MeOH1= fan1[2, :]

MeCN1= fan1[3, :]

heptane1= fan1[4, :]

H2O1= fan1[5, :]

EtOH1= fan1[6, :]

ethylacetate1= fan1[7, :]

DMSO1= fan1[8, :]

DMF1= fan1[9, :]

CHCl31= fan1[10, :]

CH2Cl21= fan1[11, :]

Toluene = fan[0, :]

THF= fan[1, :]

MeOH= fan[2, :]

MeCN= fan[3, :]

heptane= fan[4, :]

H2O= fan[5, :]

EtOH= fan[6, :]

ethylacetate= fan[7, :]
DMSO= fan[8, :]
DMF= fan[9, :]
CHCl3= fan[10, :]
CH2Cl2= fan[11, :]





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
        a=2
        for smile, y in zip(smiles,ys):
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)

            embeddings = []
            for atom in mol.GetAtoms():
                embeddings.append(atom_featurizer.encode(atom))

            embeddings = torch.tensor(embeddings,dtype=torch.float32)
            rows, cols = embeddings.shape

            toluene1 = torch.from_numpy(Toluene1)
            tHF1 = torch.from_numpy(THF1)
            meOH1 = torch.from_numpy(MeOH1)
            meCN1 = torch.from_numpy(MeCN1)
            Heptane1 = torch.from_numpy(heptane1)
            h2O1 = torch.from_numpy(H2O1)
            etOH1 = torch.from_numpy(EtOH1)
            Ethylacetate1 = torch.from_numpy(ethylacetate1)
            dMSO1 = torch.from_numpy(DMSO1)
            dMF1 = torch.from_numpy(DMF1)
            cHCl31 = torch.from_numpy(CHCl31)
            cH2Cl21 = torch.from_numpy(CH2Cl21)



            toluene1 = toluene1.repeat(rows,1)
            tHF1 = tHF1.repeat(rows,1)
            meOH1 = meOH1.repeat(rows,1)
            meCN1 = meCN1.repeat(rows,1)
            Heptane1 = Heptane1.repeat(rows,1)
            h2O1 = h2O1.repeat(rows,1)
            etOH1 = etOH1.repeat(rows,1)
            Ethylacetate1 = Ethylacetate1.repeat(rows,1)
            dMSO1 = dMSO1.repeat(rows,1)
            dMF1 = dMF1.repeat(rows,1)
            cHCl31 = cHCl31.repeat(rows,1)
            cH2Cl21 = cH2Cl21.repeat(rows,1)

            if a<= 255:
                embeddings = torch.cat((embeddings,toluene1),dim = 1)
            elif 255< a <= 498:
                embeddings = torch.cat((embeddings,tHF1),dim = 1)
            elif 498< a <= 812:
                embeddings = torch.cat((embeddings,meOH1),dim = 1)
            elif 812< a <= 1101:
                embeddings = torch.cat((embeddings,meCN1),dim = 1)
            elif 1101< a <= 1199:
                embeddings = torch.cat((embeddings,Heptane1),dim = 1)
            elif 1199< a <= 1822:
                embeddings = torch.cat((embeddings,h2O1),dim = 1)
            elif 1822< a <= 2102:
                embeddings = torch.cat((embeddings,etOH1),dim = 1)
            elif 2102< a <= 2154:
                embeddings = torch.cat((embeddings,Ethylacetate1),dim = 1)
            elif 2154< a <= 2802:
                embeddings = torch.cat((embeddings,dMSO1),dim = 1)
            elif 2802< a <= 2888:
                embeddings = torch.cat((embeddings,dMF1),dim = 1)
            elif 2888< a <= 3120:
                embeddings = torch.cat((embeddings,cHCl31),dim = 1)
            elif 3120< a :
                embeddings = torch.cat((embeddings,cH2Cl21),dim = 1)

            zeros_tensor = torch.zeros(rows, 6)

            #增加Nr
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

                #Br
                elif embeddings[i,1] == 1:
                    embeddings[i,-1] = 2.96
                    embeddings[i,-2] = 114
                    embeddings[i,-3] = 35
                    embeddings[i,-4] = 79.904
                    embeddings[i,-5] = 11.814
                    embeddings[i,-6] = 3.364

                #C
                elif embeddings[i,2] == 1:
                    embeddings[i,-1] = 2.55
                    embeddings[i,-2] = 77
                    embeddings[i,-3] = 6
                    embeddings[i,-4] = 12.011
                    embeddings[i,-5] = 11.261
                    embeddings[i,-6] = 1.595

                #Cl
                elif embeddings[i,3] == 1:
                    embeddings[i,-1] = 3.16
                    embeddings[i,-2] = 99
                    embeddings[i,-3] = 17
                    embeddings[i,-4] = 35.45
                    embeddings[i,-5] = 12.968
                    embeddings[i,-6] = 3.62
                #F
                elif embeddings[i,4] == 1:
                    embeddings[i,-1] = 3.98
                    embeddings[i,-2] = 71
                    embeddings[i,-3] = 9
                    embeddings[i,-4] = 18.998
                    embeddings[i,-5] = 17.422
                    embeddings[i,-6] = 3.40

            #Ge
                elif embeddings[i,5] == 1:
                    embeddings[i,-1] = 2.01
                    embeddings[i,-2] = 122
                    embeddings[i,-3] = 32
                    embeddings[i,-4] = 72.63
                    embeddings[i,-5] = 7.90
                    embeddings[i,-6] = 1.23


                #H
                elif embeddings[i,6] == 1:
                    embeddings[i,-1] = 2.20
                    embeddings[i,-2] = 37
                    embeddings[i,-3] = 1
                    embeddings[i,-4] = 1.008
                    embeddings[i,-5] = 13.598
                    embeddings[i,-6] = 0.755


                #I
                elif embeddings[i,7] == 1:
                    embeddings[i,-1] = 2.66
                    embeddings[i,-2] = 133
                    embeddings[i,-3] = 53
                    embeddings[i,-4] = 126.9
                    embeddings[i,-5] = 10.451
                    embeddings[i,-6] = 3.060

                 #N
                elif embeddings[i,8] == 1:
                    embeddings[i,-1] = 3.04
                    embeddings[i,-2] = 75
                    embeddings[i,-3] = 7
                    embeddings[i,-4] = 14.007
                    embeddings[i,-5] = 14.534
                    embeddings[i,-6] = 0.07

                 #Na
                elif embeddings[i,9] == 1:
                    embeddings[i,-1] = 0.93
                    embeddings[i,-2] = 154
                    embeddings[i,-3] = 11
                    embeddings[i,-4] = 22.99
                    embeddings[i,-5] = 5.139
                    embeddings[i,-6] = 0.547

                 #O
                elif embeddings[i,10] == 1:
                    embeddings[i,-1] = 3.44
                    embeddings[i,-2] = 73
                    embeddings[i,-3] = 8
                    embeddings[i,-4] = 15.999
                    embeddings[i,-5] = 13.618
                    embeddings[i,-6] = 1.46
                 #P
                elif embeddings[i,11] == 1:
                    embeddings[i,-1] = 2.19 #电负性
                    embeddings[i,-2] = 106  #共价半径
                    embeddings[i,-3] = 15   #原子序数
                    embeddings[i,-4] = 30.974 #原子质量
                    embeddings[i,-5] = 10.487 #第一电离能
                    embeddings[i,-6] = 0.75 #电子亲合能


                #S
                elif embeddings[i,12] == 1:
                    embeddings[i,-1] = 2.58 #电负性
                    embeddings[i,-2] = 102  #共价半径
                    embeddings[i,-3] = 16   #原子序数
                    embeddings[i,-4] = 32.06 #原子质量
                    embeddings[i,-5] = 10.36 #第一电离能
                    embeddings[i,-6] = 2.07 #电子亲合能

                #Se
                elif embeddings[i,13] == 1:
                    embeddings[i,-1] = 2.55 #电负性
                    embeddings[i,-2] = 116  #共价半径
                    embeddings[i,-3] = 34   #原子序数
                    embeddings[i,-4] = 78.971 #原子质量
                    embeddings[i,-5] = 9.753 #第一电离能
                    embeddings[i,-6] = 2.02 #电子亲合能
                #Si
                elif embeddings[i,14] == 1:
                    embeddings[i,-1] = 1.90 #电负性
                    embeddings[i,-2] = 111  #共价半径
                    embeddings[i,-3] = 14   #原子序数
                    embeddings[i,-4] = 28.085 #原子质量
                    embeddings[i,-5] = 8.151 #第一电离能
                    embeddings[i,-6] = 1.385 #电子亲合能

                #Te
                elif embeddings[i,15] == 1:
                    embeddings[i,-1] = 2.1 #电负性
                    embeddings[i,-2] = 135  #共价半径
                    embeddings[i,-3] = 52   #原子序数
                    embeddings[i,-4] = 127.6 #原子质量
                    embeddings[i,-5] = 9.010 #第一电离能
                    embeddings[i,-6] = 1.971 #电子亲合能
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
            rows, cols = edge_attr.shape
            toluene = torch.from_numpy(Toluene)
            tHF = torch.from_numpy(THF)
            meOH = torch.from_numpy(MeOH)
            meCN = torch.from_numpy(MeCN)
            Heptane = torch.from_numpy(heptane)
            h2O = torch.from_numpy(H2O)
            etOH = torch.from_numpy(EtOH)
            Ethylacetate = torch.from_numpy(ethylacetate)
            dMSO = torch.from_numpy(DMSO)
            dMF = torch.from_numpy(DMF)
            cHCl3 = torch.from_numpy(CHCl3)
            cH2Cl2 = torch.from_numpy(CH2Cl2)



            toluene = toluene.repeat(rows,1)
            tHF = tHF.repeat(rows,1)
            meOH = meOH.repeat(rows,1)
            meCN = meCN.repeat(rows,1)
            Heptane = Heptane.repeat(rows,1)
            h2O = h2O.repeat(rows,1)
            etOH = etOH.repeat(rows,1)
            Ethylacetate = Ethylacetate.repeat(rows,1)
            dMSO = dMSO.repeat(rows,1)
            dMF = dMF.repeat(rows,1)
            cHCl3 = cHCl3.repeat(rows,1)
            cH2Cl2 = cH2Cl2.repeat(rows,1)

            if a<= 255:
                edge_attr = torch.cat((edge_attr,toluene),dim = 1)
            elif 255< a <= 498:
                edge_attr = torch.cat((edge_attr,tHF),dim = 1)
            elif 498< a <= 812:
                edge_attr = torch.cat((edge_attr,meOH),dim = 1)
            elif 812< a <= 1101:
                edge_attr = torch.cat((edge_attr,meCN),dim = 1)
            elif 1101< a <= 1199:
                edge_attr = torch.cat((edge_attr,Heptane),dim = 1)
            elif 1199< a <= 1822:
                edge_attr = torch.cat((edge_attr,h2O),dim = 1)
            elif 1822< a <= 2102:
                edge_attr = torch.cat((edge_attr,etOH),dim = 1)
            elif 2102< a <= 2154:
                edge_attr = torch.cat((edge_attr,Ethylacetate),dim = 1)
            elif 2154< a <= 2802:
                edge_attr = torch.cat((edge_attr,dMSO),dim = 1)
            elif 2802< a <= 2888:
                edge_attr = torch.cat((edge_attr,dMF),dim = 1)
            elif 2888< a <= 3120:
                edge_attr = torch.cat((edge_attr,cHCl3),dim = 1)
            elif 3120< a :
                edge_attr = torch.cat((edge_attr,cH2Cl2),dim = 1)
            a = a+1


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
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                           [train_size,valid_size, test_size]
                                                                           ,generator=torch.Generator().manual_seed(99))
print(train_size)
print(valid_size)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# for i in train_loader:
#     print(i.edge_attr.shape)

class GATv2ConvNet(nn.Module):
    def __init__(self, edge_feature_dim_1,edge_feature_dim_2, edge_hidden_dim):
        super(GATv2ConvNet,self).__init__()
        #第一层
        self.a1 = nn.Linear(6, 6)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.m1 = nn.Linear(25, 25)
        self.e1 = nn.Linear(10,10)
        self.e2 = nn.Linear(5,5)
        self.m21 = nn.Linear(66, 32)

        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        edge_network1 = nn.Sequential(
            nn.Linear(10, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(edge_hidden_dim, 31 * 64)
        )
        self.nnconv1 = NNConv(31, 64, edge_network1, aggr="mean")

        # 第二层

        edge_network2 = nn.Sequential(
            nn.Linear(edge_feature_dim_1, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(edge_hidden_dim, 64 * 64)
        )
        self.nnconv2 = NNConv(64, 64, edge_network2, aggr="mean")
        edge_network3 = nn.Sequential(
            nn.Linear(edge_feature_dim_1, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(edge_hidden_dim,64 * 128)
        )
        self.nnconv3 = NNConv(64, 128, edge_network3, aggr="mean")
        edge_network3x = nn.Sequential(
            nn.Linear(edge_feature_dim_1, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(edge_hidden_dim,128 * 128)
        )
        self.nnconv3x = NNConv(128, 128, edge_network3x, aggr="mean")
        edge_network4 = nn.Sequential(
            nn.Linear(10, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(edge_hidden_dim,160 * 160)
        )
        self.nnconv4 = NNConv(160, 160, edge_network4, aggr="mean")
        edge_network5 = nn.Sequential(
            nn.Linear(15, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(edge_hidden_dim,128 * 64)
        )
        self.nnconv5 = NNConv(128, 64, edge_network5, aggr="mean")

        self.set2set = Set2Set(160, processing_steps=2)
        self.set2set2 = Set2Set(64, processing_steps=2)
        self.fc2 = nn.Linear(2*224, 512)
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
        x2 = x[:, 0:25]
        x2 = self.m1(x2)
        x2 = self.relu(x2)
        x3 = x[:, 25:-6]
        x3 = self.m21(x3)
        x3 = self.relu(x3)
        x = torch.cat((x1, x2), dim = 1)
        edge_attr_1 = edge_attr[:, 0:10]
        edge_attr_2 = edge_attr[:, 10:]
        edge_attr_1 = self.e1(edge_attr_1)
        edge_attr_1 = self.relu(edge_attr_1)
        edge_attr_2 = self.e2(edge_attr_2)
        edge_attr_2 = self.relu(edge_attr_2)

        x = self.nnconv1(x, edge_index, edge_attr_1)
        x = self.relu(x)
        x = self.nnconv2(x, edge_index, edge_attr_1)
        x = self.relu(x)
        x = self.nnconv3(x, edge_index, edge_attr_1)
        x = self.relu(x)
        x = self.nnconv3x(x, edge_index, edge_attr_1)
        x = self.relu(x)
        c = torch.cat((x,x3), dim = 1)
        xx = c

        '''edge_attr_1 = self.e1(edge_attr_1)
        edge_attr_1 = self.relu(edge_attr_1)
        edge_attr_2 = self.e2(edge_attr_2)
        edge_attr_2 = self.relu(edge_attr_2)'''
        xx = self.nnconv4(xx, edge_index, edge_attr_1)
        edge_attr = torch.cat((edge_attr_1, edge_attr_2), dim = 1)
        x = self.nnconv5(x, edge_index, edge_attr)

        xx = self.set2set(xx, batch)
        x = self.set2set2(x, batch)
        x = torch.cat((xx, x), dim = 1)
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


edge_feature_dim_1, edge_feature_dim_2, edge_hidden_dim = 10 , 5 , 32

# conv = NNConvNet(node_feature_dim, edge_feature_dim, edge_hidden_dim)
#
# for i in train_loader :
#     print(conv(i))

num = 1000
lr = 0.001#0.001已测

model = GATv2ConvNet(edge_feature_dim_1, edge_feature_dim_2, edge_hidden_dim)
model = model.cuda()
#print(model)

optimizer = torch.optim.Adam(model.parameters(),lr)
criterion = torch.nn.MSELoss()
criterion = criterion.cuda()

#loss
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
        out = 1000*out
        y = 1000*y
        train_preds.extend(out.detach().cpu().numpy())
        train_trues.extend(y.detach().cpu().numpy())

    mae = mean_absolute_error(train_trues,train_preds)
    r2 = r2_score(train_trues,train_preds)
    rmse = sqrt(mean_squared_error(train_trues,train_preds))
    mse = mean_squared_error(train_trues,train_preds)
    mre = mean_absolute_percentage_error(train_trues,train_preds)
    train_mae.append(mae)
    train_r2.append(r2)
    print("MAE OF TRAIN: {} ,RMSE : {} ,MRE : {} ,R2: {} ".format(mae,rmse,mre,r2))

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

            y = 1000 * y
            outputs = 1000 * outputs
            val_preds.extend(outputs.detach().cpu().numpy())
            val_trues.extend(y.detach().cpu().numpy())
        mae = mean_absolute_error(val_trues,val_preds)
        r2 = r2_score( val_trues,val_preds)
        rmse = sqrt(mean_squared_error(val_trues,val_preds))
        mse = mean_squared_error(val_trues,val_preds)
        mre = mean_absolute_percentage_error(val_trues,val_preds)
        test_mae.append(mae)
        test_r2.append(r2)
        print("MAE OF test: {} ,RMSE : {} ,MRE : {} ,R2: {} ".format(mae,rmse,mre,r2))


pd.DataFrame(train_preds).to_csv('sample1.csv')
pd.DataFrame(train_trues).to_csv('sample2.csv')
pd.DataFrame(val_preds).to_csv('sample3.csv')
pd.DataFrame(val_trues).to_csv('sample4.csv')
