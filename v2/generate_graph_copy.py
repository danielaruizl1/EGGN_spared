import numpy as np
import torch
import os
from collections import namedtuple
import torch_geometric
import sys
sys.path.insert(0, "../")
from v1.dataset import TxPDataset
from v1.main import KFOLD
from tqdm import tqdm

#Path to ST repository
import sys
sys.path.append('../../ST')
from utils import *

save_folder = "10xpro_st"
size = 256
numk = 6
mdim = 512
index_path = "index"
emb_path = "exemplars_st"
set = "train"
parser_ST = get_main_parser()
args_ST = parser_ST.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

if args_ST.dataset == "V1_Breast_Cancer_Block_A":
    dataset = get_dataset_from_args(args=args_ST)
    # Declare train and test loaders
    train_dl,val_dl,_ = dataset.get_pretrain_dataloaders(layer=args_ST.prediction_layer,batch_size = args_ST.batch_size, shuffle = args_ST.shuffle, use_cuda = use_cuda)
    dataloaders = {"trainVisium":train_dl, "valVisium":val_dl}

# Get stnet dataset from the values defined in args
elif args_ST.dataset == "stnet_dataset":
    dataset = get_dataset_from_args(args=args_ST)
    # Declare train, test and val dataloaders
    train_dl, val_dl, test_dl = dataset.get_pretrain_dataloaders(layer=args_ST.prediction_layer,batch_size = args_ST.batch_size, shuffle = args_ST.shuffle, use_cuda = use_cuda)
    dataloaders = {"trainSTnet":train_dl, "valSTnet":val_dl, "testSTnet":test_dl}

fold=0
save_name = save_folder + "/" + str(fold)
os.makedirs(save_name,exist_ok=True)
foldername = f"{save_name}/graphs_{numk}"
os.makedirs(foldername, exist_ok=True)                    

for exemplar_name,dataloader in dataloaders.items():
    embs = torch.load(f"{emb_path}/{exemplar_name}.pt")
    index = np.load(f"{emb_path}/{fold}/{index_path}/{exemplar_name}.npy")
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

    torch.save(data, f"{foldername}/{exemplar_name}.pt")