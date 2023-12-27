import argparse
import os
from model_sepal import HeteroGNN
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train_EGGN_sepal import TrainerModel
import glob
import torch_geometric
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from datetime import datetime

cudnn.benchmark = True
use_cuda = torch.cuda.is_available()

# Add argparse
parser = argparse.ArgumentParser(description="Arguments for training EGGN")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
parser.add_argument("--prediction_layer", type=str, default="c_d_log1p", help="Layer to use for prediction")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--num_cores', type=int, default=12, help='Number of cores')
parser.add_argument('--numk', type=int, default=6, help='Number of k')
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps")
parser.add_argument("--val_interval", type=int, default=1, help="Validation interval")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--verbose_step", type=int, default=10, help="Verbose step")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--mdim", type=int, default=512, help="Dimension of the message")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--optim_metric", type=str, default="MSE", help="Metric to optimize")
args = parser.parse_args()

dataset_config_path = os.path.join("spared","configs",args.dataset+".json")

with open(dataset_config_path, 'r') as f:
    dataset_config = json.load(f)

def load_datasets(args, graph_path):

    all_pts = glob.glob(f"{graph_path}/*.pt")
    datasets= {}

    for i in all_pts:
        if i.endswith(f"{args.dataset}.pt"):
            graph = torch.load(i)
            dataset_name=i.split("/")[-1]
            datasets[dataset_name[:-3]]=[graph]

    return datasets

# Initialize wandb
exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="EGGN_sota", name=exp_name, config=vars(args))

graph_path = f"graphs/{args.dataset}"
num_genes = dataset_config["top_moran_genes"]
cwd = os.getcwd()

def write(director,name,*string):
    string = [str(i) for i in string]
    string = " ".join(string)
    with open(os.path.join(director,name),"a") as f:
        f.write(string + "\n")
        
save_dir = f"results/{args.dataset}"
print = partial(write,cwd,save_dir + f"/log_{args.dataset}") 
    
os.makedirs(save_dir, exist_ok= True)

datasets = load_datasets(args, graph_path)
train_loader = torch_geometric.loader.DataLoader(datasets[f"train_{args.dataset}"],batch_size=1)
val_loader = torch_geometric.loader.DataLoader(datasets[f"val_{args.dataset}"],batch_size=1) 

model = HeteroGNN(args.num_layers, args.mdim, num_genes)
CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'dataset','verbose_step', 'weight_decay', 'store_dir', 'opt_metric', 'num_genes', 'max_steps'])
config = CONFIG(args.lr, print, args.dataset, args.verbose_step, args.weight_decay, save_dir, args.optim_metric, num_genes, args.max_steps)

model = TrainerModel(config, model)

plt = pl.Trainer(num_nodes=1, devices = args.gpus, max_steps = args.max_steps, val_check_interval = args.val_interval, 
                 check_val_every_n_epoch = None, strategy="ddp",
                 callbacks=[ModelCheckpoint(dirpath="checkpoints/"+wandb.run.name, monitor='val_loss')], logger = False)

plt.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)

if f"test_{args.dataset}" in datasets.keys():
    test_loader = torch_geometric.loader.DataLoader(datasets[f"test_{args.dataset}"],batch_size=1)
    plt.test(model, dataloaders=test_loader, ckpt_path=glob.glob(os.path.join("checkpoints",exp_name,"*.ckpt"))[0])
    
wandb.finish()



