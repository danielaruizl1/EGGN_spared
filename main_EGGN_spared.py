import argparse
import os
from model_spared import HeteroGNN
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train_EGGN_spared import TrainerModel
import glob
import torch_geometric
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from datetime import datetime
from spared.spared_datasets import get_dataset
import pandas as pd
import sys
from lightning.pytorch import seed_everything

# Set manual seeds and get cuda
seed_everything(42, workers=True)
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()

# Declare device
device = torch.device("cuda" if use_cuda else "cpu")

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
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps")
parser.add_argument("--val_interval", type=int, default=1, help="Validation interval")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument('--use_optimal_lr', type=str2bool, default=False, help='Whether or not to use the optimal learning rate in csv for the dataset.')
parser.add_argument("--verbose_step", type=int, default=10, help="Verbose step")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--mdim", type=int, default=512, help="Dimension of the message")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--optim_metric", type=str, default="MSE", help="Metric to optimize")
parser.add_argument("--patches_key", type=str, default="patches_scale_1.0", help="Key of the patches in the dataset")
parser.add_argument("--graph_radius", type=float, default=1000, help="Graph radius")
parser.add_argument("--train", type=str2bool, default=True, help="Train or load the model")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint")
args = parser.parse_args()

spared_path = next((path for path in sys.path if 'spared' in path), None)
dataset_config_path = os.path.join(spared_path,"configs",args.dataset+".json")

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

# Obtain optimal lr depending on the dataset
if args.use_optimal_lr:
    optimal_models_directory_path =  'wandb_runs_csv/optimal_lr_eggn_ctlog1p.csv'
    optimal_lr_df = pd.read_csv(optimal_models_directory_path, sep=";")
    optimal_lr = float(optimal_lr_df[optimal_lr_df['Dataset'] == args.dataset]['eggn'])
    args.lr = optimal_lr
    print(f'Optimal lr for {args.dataset} is {optimal_lr}')

# Initialize wandb
exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="eggn_spared", name=exp_name, config=vars(args))

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
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/"+wandb.run.name, monitor='val_loss')

plt = pl.Trainer(num_nodes=1, devices = args.gpus, max_steps = args.max_steps, val_check_interval = args.val_interval, 
                 check_val_every_n_epoch = None, strategy="ddp",
                 callbacks=[checkpoint_callback], logger = False)

if args.train:
    plt.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    checkpoint_path= checkpoint_callback.best_model_path
else:
    checkpoint_path = args.checkpoint_path

if f"test_{args.dataset}" in datasets.keys():
    test_loader = torch_geometric.loader.DataLoader(datasets[f"test_{args.dataset}"],batch_size=1)
    plt.test(model, dataloaders=test_loader, ckpt_path=checkpoint_path)

# Load the best model after training
checkpoint = torch.load(checkpoint_path)

# Modify state dict to have the same keys
original_state_dict = checkpoint['state_dict']
new_state_dict = {}
for key in original_state_dict:
    # Remove the 'model.' prefix from each key
    new_key = key.replace('model.', '')
    new_state_dict[new_key] = original_state_dict[key]

# Update the state dictionary in the checkpoint
checkpoint['state_dict'] = new_state_dict

# Load the model
model = HeteroGNN(args.num_layers, args.mdim, num_genes)
model.load_state_dict(checkpoint['state_dict'])

# Get dataset from the values defined in args
dataset = get_dataset(args.dataset, visualize=False)

def get_predictions(model)->None:
    
    # Get complete dataloader
    dataloaders = {'train':train_loader, 'val':val_loader}
    if f"test_{args.dataset}" in datasets.keys():
        dataloaders['test'] = test_loader

    # Define global variables
    glob_expression_pred = None
    glob_ids = None

    # Set model to eval mode
    model.eval()

    # Get complete predictions
    with torch.no_grad():
        for set_name in dataloaders.keys():
            for data in dataloaders[set_name]:
                # Splitting into x_dict and edge_index_dict
                x_dict = {
                    'window': data['window'].x.to(device),
                    'example': data['example'].x.to(device)
                }

                edge_index_dict = {
                    ('window', 'near', 'window'): data[('window', 'near', 'window')]['edge_index'].to(device),
                    ('example', 'refer', 'window'): data[('example', 'refer', 'window')]['edge_index'].to(device),
                    ('example', 'close', 'example'): data[('example', 'close', 'example')]['edge_index'].to(device)
                }

                expression_pred = model(x_dict, edge_index_dict)
                ids = dataset.adata.obs[dataset.adata.obs.split == set_name].index.tolist()

                # Concat batch to get global predictions and IDs
                glob_expression_pred = expression_pred if glob_expression_pred is None else torch.cat((glob_expression_pred, expression_pred))
                glob_ids = ids if glob_ids is None else glob_ids + ids

        # Handle delta prediction
        if 'deltas' in args.prediction_layer:
            mean_key = f'{args.prediction_layer}_avg_exp'.replace('deltas', 'log1p')
            means = torch.tensor(dataset.adata.var[mean_key], device=glob_expression_pred.device)
            glob_expression_pred = glob_expression_pred+means
        
        # Put complete predictions in a single dataframe
        pred_matrix = glob_expression_pred.detach().cpu().numpy()
        pred_df = pd.DataFrame(pred_matrix, index=glob_ids, columns=dataset.adata.var_names)
        pred_df = pred_df.reindex(dataset.adata.obs.index)

        # Log predictions to wandb
        wandb_df = pred_df.reset_index(names='sample')
        wandb.log({'predictions': wandb.Table(dataframe=wandb_df)})
        
        # Add layer to adata
        dataset.adata.layers[f'predictions,{args.prediction_layer}'] = pred_df

# Get global prediction layer 
get_predictions(model.to(device))

# If noisy layer, change to c_d_log1p for log final artifacts
if args.prediction_layer == 'noisy':
    preds = dataset.adata.layers['predictions,noisy']
    dataset.adata.layers['predictions,c_d_log1p'] = preds
    del dataset.adata.layers['predictions,noisy']

# Get log final artifacts
#dataset.log_pred_image()
    
wandb.finish()



