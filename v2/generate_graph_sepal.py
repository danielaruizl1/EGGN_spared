import numpy as np
import torch
import os
from collections import namedtuple
import torch_geometric
import sys
from tqdm import tqdm

# Current path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# Add path to ST repository
sepal_dir = os.path.join(parent_dir[:-4], 'SEPAL')
sys.path.append(sepal_dir)

# Import SEPAL utils
from utils import *

def get_edge(x,radius):

    edge_index = torch_geometric.nn.radius_graph(x,
            radius,
            None,
            False,
            max_num_neighbors=5,
            flow="source_to_target",
            num_workers=1,
        )
     
    return edge_index


def get_cross_edge(x):
    
    l = len(x)
    source = torch.LongTensor(range(l))
    
    op = torch.cat([i[3] for i in x]).clone()
    opy = torch.cat([i[4] for i in x]).clone().to(device=op.device)

    b,n,c= op.shape
    source = torch.repeat_interleave(source, n)
    
    ops = torch.cat((op,opy),-1).view(b*n,-1)
    ops,inverse = torch.unique(ops,dim=0, return_inverse=True)
    unique_op = ops[:,:c]
    unique_opy = ops[:,c:]
    
    edge = torch.stack((source,inverse.to(device=source.device)))
    return unique_op, unique_opy, edge

def retrive_similer(index, i):
    index = index[i]
    index = np.array([[i,j,k] for i,j,k in sorted(index, key = lambda x : float(x[0]))]) 
    index = index[-numk:]

    op_emb = []
    op_counts = []
    for _, op_name, _ in index:
        op_name = int(op_name)
        op_emb.append(embs[op_name])
        #op_counts.append(torch.tensor(dataset.data.X[op_name].todense()).squeeze())
        op_counts.append(torch.tensor(dataset.adata.X[op_name]).squeeze())

    return torch.stack(op_emb).view(numk,-1), torch.stack(op_counts).view(numk,len(op_counts[0]))    

parser_ST = get_main_parser()
args_ST = parser_ST.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Parameters
save_folder = f"graphs/{args_ST.dataset}"
emb_path = f"exemplars/{args_ST.dataset}"
numk = 6

# Create save folder if necessary
os.makedirs(os.path.join(current_dir,save_folder),exist_ok=True)     

# Get dataset from the values defined in args
dataset = get_dataset_from_args(args=args_ST)
# Declare data loaders
train_dl, val_dl, test_dl = dataset.get_pretrain_dataloaders(layer=args_ST.prediction_layer, batch_size = args_ST.batch_size, shuffle = args_ST.shuffle, use_cuda = use_cuda)
dataloaders = {f"train_{args_ST.dataset}": train_dl, f"val_{args_ST.dataset}": val_dl}
# Add test loader only if it is not None
if test_dl is not None:
    dataloaders[f"test_{args_ST.dataset}"] = test_dl     

for exemplar_name,dataloader in dataloaders.items():
    embs = torch.load(f"{current_dir}/{emb_path}/{exemplar_name}.pt")
    index = np.load(f"{current_dir}/{emb_path}/{exemplar_name}.npy")
    img_data = []
    masks = []
    for i in tqdm(range(len(dataloader.dataset))):
        pos = torch.tensor(dataloader.dataset[i].obsm["spatial"]) #torch.Size([1, 2])
        p = embs[i] #torch.Size([512])
        p = torch.unsqueeze(p,0)
        py = torch.tensor(dataloader.dataset[:].X[i]).squeeze() #torch.Size([1024])
        py = torch.unsqueeze(py,0)
        op, opy = retrive_similer(index, i) #torch.Size([6, 512]), torch.Size([6, 1024])
        op = torch.unsqueeze(op,0)
        opy = torch.unsqueeze(opy,0)
        mask = torch.tensor(dataloader.dataset[i].layers["mask"])
        masks.append(mask)
        img_data.append([pos, p, py, op, opy])

    all_img_data = torch.cat(([i[0] for i in img_data])).clone()
    window_edge = get_edge(all_img_data.type(torch.float),275)
    unique_op, unique_opy, cross_edge = get_cross_edge(img_data)
    print(window_edge.size(), unique_op.size(), unique_opy.size(), cross_edge.size())

    data = torch_geometric.data.HeteroData()
    data["window"].pos = torch.cat(([i[0] for i in img_data])).clone()
    data["window"].x = torch.cat(([i[1] for i in img_data])).clone()
    data["window"].x = data["window"].x.squeeze()
    data["window"].y = torch.cat(([i[2] for i in img_data])).clone()
    data["window"].mask = torch.cat(([i for i in masks])).clone()

    assert len(data["window"]["pos"]) == len(data["window"]["x"]) == len(data["window"]["y"]) == len(data["window"]["mask"])

    data["example"].x = torch.cat((unique_op, unique_opy),-1)
    data['window', 'near', 'window'].edge_index = window_edge
    data["example", "refer", "window"].edge_index = cross_edge[[1,0]]
    edge_index = torch_geometric.nn.knn_graph(data["example"]["x"], k=3, loop=False)
    data["example", "close", "example"].edge_index = edge_index

    torch.save(data, f"{current_dir}/{save_folder}/{exemplar_name}.pt")