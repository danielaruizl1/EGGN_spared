import argparse
import os
from model_copy import HeteroGNN
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train_copy import TrainerModel
from pytorch_lightning.plugins import DDPPlugin
from datetime import datetime
import glob
cudnn.benchmark = True
import torch_geometric
import sys
sys.path.insert(0, "../")
from v1.main import KFOLD
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

def load_datasets(args):

    all_pts = glob.glob(f"{args.graph_path}/{args.fold}/graphs_{args.numk}/*.pt")
    datasets= {}

    for i in all_pts:
        if i.endswith(f"{args.dataset}.pt"):
            graph = torch.load(i)
            dataset_name=i.split("/")[-1]
            datasets[dataset_name[:-3]]=[graph]

    return datasets

def main(args):

    # Initialize wandb
    wandb.init(project="EGN_v2", config=vars(args))

    cwd = os.getcwd()
    
    def write(director,name,*string):
        string = [str(i) for i in string]
        string = " ".join(string)
        with open(os.path.join(director,name),"a") as f:
            f.write(string + "\n")
            
    store_dir = os.path.join(args.output,str(args.fold))
    print = partial(write,cwd,args.output + f"/log_{args.dataset}") 
        
    os.makedirs(store_dir, exist_ok= True)
    
    print(args)
    
    datasets = load_datasets(args)
    train_loader = torch_geometric.loader.DataLoader(datasets[f"train{args.dataset}"],batch_size=1)
    val_loader = torch_geometric.loader.DataLoader(datasets[f"val{args.dataset}"],batch_size=1)      
    
    model = HeteroGNN(args.num_layers,args.mdim,args.num_genes)
    CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'dataset','verbose_step', 'weight_decay', 'store_dir', 'opt_metric'])
    config = CONFIG(args.lr, print, args.dataset, args.verbose_step, args.weight_decay, store_dir, args.optim_metric)
    
    model = TrainerModel(config, model)
    
    plt = pl.Trainer(max_epochs = args.epoch,num_nodes=1, gpus=args.gpus, val_check_interval = args.val_interval,
                    strategy=DDPPlugin(find_unused_parameters=False),checkpoint_callback = True,
                    callbacks=[ModelCheckpoint(dirpath="checkpoints/"+wandb.run.name, monitor='val_loss')], logger = False)

    if args.train == True:
    
        plt.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
        if args.dataset=="STnet":
            test_loader = torch_geometric.loader.DataLoader(datasets[f"test{args.dataset}"],batch_size=1)
            plt.test(model, dataloaders=test_loader, ckpt_path="best")

    else:

        if args.dataset == "Visium":
            plt.test(model, dataloaders=val_loader, ckpt_path=os.path.join("checkpoints","best_visium","*.ckpt"))
        else:
            test_loader = torch_geometric.loader.DataLoader(datasets[f"test{args.dataset}"],batch_size=1)
            plt.test(model, dataloaders=test_loader, ckpt_path=os.path.join("checkpoints","best_stnet","*.ckpt"))
    
    wandb.finish()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", required=True, type = str) #Visium or STnet
    parser.add_argument("--epoch", default = 30, type = int)
    parser.add_argument("--fold", default = 0, type = int)
    parser.add_argument("--gpus", required=True, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--val_interval", default = 1, type = float)
    parser.add_argument("--lr", required=True, type = float)
    parser.add_argument("--verbose_step", default = 10, type = int)
    parser.add_argument("--weight_decay", required=True, type = float)
    parser.add_argument("--mdim", required=True, type = int)
    parser.add_argument("--output", default = "results_st", type = str)
    parser.add_argument("--numk", required=True, type = int)
    parser.add_argument("--num_layers", required=True, type = int)
    parser.add_argument("--graph_path", required=True, type = str)
    parser.add_argument("--num_genes", default=256, type = int)
    parser.add_argument("--optim_metric", default="PCC-Gene", type = str)
    parser.add_argument("--train", default=True, type = bool)
    
    args = parser.parse_args()
    main(args)


    
