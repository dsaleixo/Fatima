import pandas as pd
import networkx as nx
import numpy as np
from biopandas.pdb import PandasPdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

from pathlib import Path




# --- Aminoácidos e one-hot ---
AA_LIST = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def one_hot_aa(res_name: str) -> np.ndarray:
    vec = np.zeros(len(AA_LIST))
    if res_name in AA_TO_IDX:
        vec[AA_TO_IDX[res_name]] = 1.0
    return vec

# --- Função para construir grafo de resíduos ---
def build_residue_graph_from_pdb(pdb_id: str, cutoff: float = 8.0) -> nx.Graph:
    ppdb = PandasPdb().fetch_pdb(pdb_id)
    df = ppdb.df['ATOM']
    df_ca = df[df['atom_name'] == 'CA']

    G = nx.Graph()
    for _, row in df_ca.iterrows():
        res_id = int(row['residue_number'])
        res_name = row['residue_name'].strip()
        coord = np.array([row['x_coord'], row['y_coord'], row['z_coord']])
        G.add_node(res_id, aa=res_name, aa_onehot=one_hot_aa(res_name), coord=coord)

    nodes = list(G.nodes(data=True))
    for i, (id1, data1) in enumerate(nodes):
        for j, (id2, data2) in enumerate(nodes):
            if i >= j:
                continue
            dist = np.linalg.norm(data1['coord'] - data2['coord'])
            if dist <= cutoff:
                G.add_edge(id1, id2, distance=dist)
    return G

# --- Função para converter grafo NetworkX -> PyG Data ---
def nx_to_pyg(id :str,G: nx.Graph, label: int) -> Data:
    x = torch.tensor([data['aa_onehot'] for _, data in G.nodes(data=True)], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    # Edge_attr opcional
    edge_attr = torch.tensor([G[u][v]['distance'] for u, v in G.edges], dtype=torch.float).unsqueeze(1)
    y = torch.tensor([label], dtype=torch.long)
    print(x.shape,edge_index.shape,edge_attr.shape)
    return Data(id=id,x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# --- GNN para classificação ---
class ProteinGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0, keepdim=True)  # pooling global
        x = self.fc(x)
        return x

# --- Pipeline principal ---
class BuildDatas:
    @staticmethod
    def BuildMdrDB_ResidueGraph():
        df = pd.read_csv("MdrDB/data/MdrDB_CoreSet_release_v1.0.2022.tsv", sep="\t") 
     
        cont = 0
        for idx, row in df.iterrows():

            cont+=1
            pdb_id = row['SAMPLE_PDB_ID']

            path = Path(f"datas/MdrDB_residueGraph/{row['SAMPLE_ID']}.pt")

            if path.exists():
                print(f"{cont} PDB {pdb_id}: already processed")
                continue


           

            impact = 1 if row['DDG.EXP'] > 0 else 0  # exemplo: deleterious se ΔΔG>0
            try:
                G = build_residue_graph_from_pdb(pdb_id)
                data = nx_to_pyg(row['SAMPLE_ID'],G, label=impact)
                torch.save(data, f"datas/MdrDB_residueGraph/{row['SAMPLE_ID']}.pt")  
                print(f"{cont} PDB {pdb_id}: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
            except Exception as e:
                print(f"Erro ao processar {pdb_id}: {e}")
        return 

if __name__ == "__main__":
    # Lê MdrDB e constrói dataset
    BuildDatas.BuildMdrDB_ResidueGraph()
    