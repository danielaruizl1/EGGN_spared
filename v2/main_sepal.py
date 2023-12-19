import argparse
import os
from model_sepal import HeteroGNN
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train_sepal import TrainerModel
from pytorch_lightning.plugins import DDPPlugin
from datetime import datetime
import glob
cudnn.benchmark = True
import torch_geometric
import sys
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

# Current path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# Add path to ST repository
sepal_dir = os.path.join(parent_dir[:-4], 'SEPAL')
sys.path.append(sepal_dir)

# Import SEPAL utils
from utils import *

parser_ST = get_main_parser()
args_ST = parser_ST.parse_args()
use_cuda = torch.cuda.is_available()

config_path = os.path.join("SEPAL","configs","datasets",args_ST.dataset+".json")

with open(config_path, 'r') as f:
    dataset_config = json.load(f)

def load_datasets(args_ST, graph_path):

    all_pts = glob.glob(f"{graph_path}/*.pt")
    datasets= {}

    for i in all_pts:
        if i.endswith(f"{args_ST.dataset}.pt"):
            graph = torch.load(i)
            dataset_name=i.split("/")[-1]
            datasets[dataset_name[:-3]]=[graph]

    return datasets

# Initialize wandb
wandb.init(project="EGN_v2", config=vars(args_ST))

graph_path = f"{current_dir}/graphs/{args_ST.dataset}"
num_genes = dataset_config["top_moran_genes"]
cwd = os.getcwd()

# Parameters
epochs = 50
gpus = 1
val_interval = 0.8
lr = 5e-4
verbose_step = 10
weight_decay = 1e-4
mdim = 512
num_layers = 4
optim_metric = "MSE"

def write(director,name,*string):
    string = [str(i) for i in string]
    string = " ".join(string)
    with open(os.path.join(director,name),"a") as f:
        f.write(string + "\n")
        
save_dir = f"{current_dir}/results/{args_ST.dataset}"
print = partial(write,cwd,save_dir + f"/log_{args_ST.dataset}") 
    
os.makedirs(save_dir, exist_ok= True)

datasets = load_datasets(args_ST, graph_path)
train_loader = torch_geometric.loader.DataLoader(datasets[f"train_{args_ST.dataset}"],batch_size=1)
val_loader = torch_geometric.loader.DataLoader(datasets[f"val_{args_ST.dataset}"],batch_size=1) 

model = HeteroGNN(num_layers,mdim,num_genes)
CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'dataset','verbose_step', 'weight_decay', 'store_dir', 'opt_metric'])
config = CONFIG(lr, print, args_ST.dataset, verbose_step, weight_decay, save_dir, optim_metric)

model = TrainerModel(config, model, num_genes)

plt = pl.Trainer(max_epochs = epochs, num_nodes=1, gpus=gpus, val_check_interval = val_interval,
                strategy=DDPPlugin(find_unused_parameters=False),checkpoint_callback = True,
                callbacks=[ModelCheckpoint(dirpath="checkpoints/"+wandb.run.name, monitor='val_loss')], logger = False)

plt.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)

if f"test_{args_ST.dataset}" in datasets.keys():
    test_loader = torch_geometric.loader.DataLoader(datasets[f"test_{args_ST.dataset}"],batch_size=1)
    plt.test(model, dataloaders=test_loader, ckpt_path=os.path.join("checkpoints",f"best_{args_ST.dataset}","*.ckpt"))
    
wandb.finish()



