from metrics import get_metrics
from tqdm import tqdm
import numpy as np
import random
import torch

#Path to ST repository
import sys
sys.path.append('../../ST')
from utils import *

parser_ST = get_main_parser()
args_ST = parser_ST.parse_args()
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
emb_path = "exemplars_st"
index_path = "index"
fold=0

# Get train and test visium datasets from the values defined in args
if args_ST.dataset == "visium":
    train_dataset, test_dataset = get_datasets_from_args(args=args_ST)
    train_dl = train_dataset.get_dataloader(batch_size = args_ST.batch_size, shuffle = args_ST.shuffle, use_cuda = use_cuda)
    test_dl  = test_dataset.get_dataloader( batch_size = args_ST.batch_size, shuffle = args_ST.shuffle, use_cuda = use_cuda)
    dataloaders = {"trainVisium":train_dl, "testVisium":test_dl}

# Get stnet dataset from the values defined in args
elif args_ST.dataset == "stnet_dataset":
    dataset = get_datasets_from_args(args=args_ST)
    # Declare train, test and val dataloaders
    train_dl, val_dl, test_dl = dataset.get_dataloaders(batch_size = args_ST.batch_size, shuffle = args_ST.shuffle, use_cuda = use_cuda)
    dataloaders = {"trainSTnet":train_dl, "valSTnet":val_dl, "testSTnet":test_dl}

def retrive_similer(index, i, numk, embs):
    index = index[i]
    index = np.array([[i,j,k] for i,j,k in sorted(index, key = lambda x : float(x[0]))]) 
    index = index[-numk:]

    op_emb = []
    op_counts = []
    for _, op_name, _ in index:
        op_name = int(op_name)
        op_emb.append(embs[op_name])
        #op_counts.append(torch.tensor(dataset.data.X[op_name].todense()).squeeze())
        op_counts.append(torch.tensor(dataset.data.X[op_name]).squeeze())

    return torch.stack(op_emb).view(numk,-1), torch.stack(op_counts).view(numk,len(op_counts[0]))   

def obtain_metrics(exemplar_name, dataloader, mode):

    embs = torch.load(f"{emb_path}/{exemplar_name}.pt")
    index = np.load(f"{emb_path}/{fold}/{index_path}/{exemplar_name}.npy")
    gt_mat = torch.Tensor().to(device)
    pred_mat = torch.Tensor().to(device)
    mean_expression = torch.mean(torch.tensor(dataloader.dataset[:].X).squeeze(),axis=0)
    mean_expression = torch.unsqueeze(mean_expression,0).to(device)
    
    for i in tqdm(range(len(dataloader.dataset))):
        py = torch.tensor(dataloader.dataset[:].X[i]).squeeze() #torch.Size([1024])
        py = torch.unsqueeze(py,0).to(device)
        op, opy = retrive_similer(index, i, 6, embs) #torch.Size([6, 512]), torch.Size([6, 1024])
        opy = torch.unsqueeze(opy,0).to(device)     
        gt_mat = torch.cat((gt_mat,py),0).to(device)

        # Most similar sample
        if mode == "NN":
            pred = torch.unsqueeze(opy[0][0],0)
            pred_mat = torch.cat((pred_mat,pred),0)

        # Second most similar sample
        elif mode == "2NN":
            pred = torch.unsqueeze(opy[0][1],0)
            pred_mat = torch.cat((pred_mat,pred),0)

        # Random sample
        elif mode == "random":
            j = random.randint(1,len(dataloader)-1)
            p2 = embs[j] #torch.Size([512])
            p2 = torch.unsqueeze(p2,0)
            py2 = torch.tensor(dataloader.dataset[:].X[j]).squeeze() #torch.Size([1024])
            py2 = torch.unsqueeze(py2,0)
            pred_mat = torch.cat((pred_mat,py2),0)

        # Mean of the dataset    
        elif mode == "mean":
            pred_mat = torch.cat((pred_mat,mean_expression),0)
        
    metrics=get_metrics(gt_mat, pred_mat)
    return metrics

for exemplar_name,dataloader in dataloaders.items():
    print(exemplar_name)
    metrics_NN = obtain_metrics(exemplar_name, dataloader, "NN")
    print("NN")
    print(metrics_NN)
    metrics_2NN = obtain_metrics(exemplar_name, dataloader, "2NN")
    print("2NN")
    print(metrics_2NN)
    metrics_random = obtain_metrics(exemplar_name, dataloader, "random")
    print("random")
    print(metrics_random)
    metrics_mean = obtain_metrics(exemplar_name, dataloader, "mean")  
    print("mean")
    print(metrics_mean)
    
