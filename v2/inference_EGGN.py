import argparse
from spared.datasets import get_dataset
from main_EGGN_spared import get_predictions
import torch
import pandas as pd
from EGGN_sepal.v2.model_spared import HeteroGNN
import os
import json

# Add argparse
parser = argparse.ArgumentParser(description="Arguments for training HisToGene")
parser.add_argument("--dataset", type=str, default="10xgenomic_human_breast_cancer", help="Dataset to use")
parser.add_argument("--checkpoint_path", type=str, help="Checkpoint path")
args = parser.parse_args()

# Declare device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Get dataset from the values defined in args
dataset = get_dataset(args.dataset)

# Get dataset config
dataset_config_path = os.path.join("spared","configs",args.dataset+".json")

with open(dataset_config_path, 'r') as f:
    dataset_config = json.load(f)

num_genes = dataset_config["top_moran_genes"]

# Load the best model after training
checkpoint = torch.load(args.checkpoint_path)

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

# Get global prediction layer 
dataset = get_predictions(model.to(device), dataset)

# Save predictions
dataset.adata.write_h5ad(f'sota_predictions/{args.dataset}/eggn_adata_pred.h5ad')