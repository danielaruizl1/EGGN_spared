import numpy as np
import torch
import os
import torch_geometric
from tqdm import tqdm
from spared.datasets import get_dataset
import argparse
import json

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

# Add argparse
parser = argparse.ArgumentParser(description="Arguments for training EGGN")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
parser.add_argument("--prediction_layer", type=str, default="c_d_log1p", help="Layer to use for prediction")
parser.add_argument('--batch_size_dataloader', type=int, default=64, help='Batch size')
parser.add_argument('--num_cores', type=int, default=12, help='Number of cores')
parser.add_argument('--numk', type=int, default=6, help='Number of k')
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
parser.add_argument("--max_steps", type=int, default=100, help="Max steps")
parser.add_argument("--val_interval", type=float, default=0.8, help="Validation interval")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument('--use_optimal_lr', type=str2bool, default=False, help='Whether or not to use the optimal learning rate in csv for the dataset.')
parser.add_argument("--verbose_step", type=int, default=10, help="Verbose step")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--mdim", type=int, default=512, help="Dimension of the message")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--optim_metric", type=str, default="MSE", help="Metric to optimize")
parser.add_argument("--patches_key", type=str, default="patches_scale_1.0", help="Key of the patches in the dataset")
parser.add_argument("--graph_radius", type=float, default=1000, help="Graph radius")
args = parser.parse_args()

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
    index = index[-args.numk:]

    op_emb = []
    op_counts = []
    for _, op_name, _ in index:
        op_name = int(op_name)
        op_emb.append(embs_train[op_name])
        op_counts.append(torch.tensor(dataset.adata.X[op_name]).squeeze())

    return torch.stack(op_emb).view(args.numk,-1), torch.stack(op_counts).view(args.numk,len(op_counts[0]))    

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Paths
save_folder = f"graphs/{args.dataset}"
emb_path = f"exemplars/{args.dataset}"

# Create save folder if necessary
os.makedirs(save_folder, exist_ok=True)     

# Get dataset from the values defined in args
dataset = get_dataset(args.dataset)

# Declare data loaders
train_dl, val_dl, test_dl = dataset.get_pretrain_dataloaders(layer=args.prediction_layer, 
                                                             batch_size = args.batch_size_dataloader, 
                                                             use_cuda = use_cuda)
dataloaders = {f"train_{args.dataset}": train_dl, f"val_{args.dataset}": val_dl}
# Add test loader only if it is not None
if test_dl is not None:
    dataloaders[f"test_{args.dataset}"] = test_dl    

embs_train = torch.load(f"{emb_path}/train_{args.dataset}.pt") 

for exemplar_name, dataloader in dataloaders.items():
    embs = torch.load(f"{emb_path}/{exemplar_name}.pt")
    index = np.load(f"{emb_path}/{exemplar_name}.npy")
    img_data = []
    masks = []
    for i in tqdm(range(len(dataloader.dataset))):
        pos = torch.tensor(dataloader.dataset[i].obsm["spatial"]) #torch.Size([1, 2])
        p = embs[i] #torch.Size([512])
        p = torch.unsqueeze(p,0)
        py = torch.tensor(dataloader.dataset[:].X[i]).squeeze() #torch.Size([num_genes])
        py = torch.unsqueeze(py,0)
        op, opy = retrive_similer(index, i) #torch.Size([6, 512]), torch.Size([6, num_genes])
        op = torch.unsqueeze(op,0)
        opy = torch.unsqueeze(opy,0)
        mask = torch.tensor(dataloader.dataset[i].layers["mask"])
        masks.append(mask)
        img_data.append([pos, p, py, op, opy])

    all_img_data = torch.cat(([i[0] for i in img_data])).clone() 
    window_edge = get_edge(all_img_data.type(torch.float),args.graph_radius)
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

    torch.save(data, f"{save_folder}/{exemplar_name}.pt")