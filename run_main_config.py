from spared.spared_datasets import get_dataset
from spared.denoising import spackle_cleaner
import subprocess
import argparse
import torch
import json
import os

def run_egn():
    
    # Auxiliary function to use booleans in parser
    str2bool = lambda x: (str(x).lower() == 'true')
    str2intlist = lambda x: [int(i) for i in x.split(',')]
    str2floatlist = lambda x: [float(i) for i in x.split(',')]
    str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

    # Get parsed the path of the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="10xgenomic_human_breast_cancer", help='Dataset name')
    parser.add_argument('--lr', type=float, default=0.002154, help='Learning rate')
    parser.add_argument('--use_optimal_lr', type=str2bool, default=False, help='Whether or not to use the optimal learning rate in csv for the dataset.')
    parser.add_argument('--prediction_layer', type=str, default="c_t_log1p", help='Layer to use for prediction')
    parser.add_argument('--train_config', type=str, default="train_EGGN_config.json", help='Config file path with train hyperparameters')
    parser.add_argument("--train", type=str2bool, default=True, help="Train or load the model")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint")
    parser.add_argument("--original_index", type=str2bool, default=False, help="Whether to use the original index")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get dataset according to the dataset name passed in args
    dataset = get_dataset(args.dataset, visualize=False)

    # Denoise the dataset if necessary
    if (args.prediction_layer == 'c_t_log1p') and (not args.prediction_layer in dataset.adata.layers):
        dataset.adata, _  = spackle_cleaner(adata=dataset.adata, dataset=args.dataset, from_layer="c_d_log1p", to_layer="c_t_log1p", device=device)
        # Replace current adata.h5ad file for the one with the completed data layer.
        dataset.adata.write_h5ad(os.path.join(dataset.dataset_path, "adata.h5ad"))

    elif (args.prediction_layer == "c_t_deltas") and (not args.prediction_layer in dataset.adata.layers):
        dataset.adata, _  = spackle_cleaner(adata=dataset.adata, dataset=args.dataset, from_layer="c_d_deltas", to_layer="c_t_deltas", device=device)
        # Replace current adata.h5ad file for the one with the completed data layer.
        dataset.adata.write_h5ad(os.path.join(dataset.dataset_path, "adata.h5ad"))

    commands = [
        ['python', 'build_exemplar_spared.py'],
        ['python', 'generate_graph_spared.py'],
        ['python', 'main_EGGN_spared.py']
    ]

    # Upload the config file 
    with open(args.train_config) as f:
        train_config = json.load(f)   
    
    # Add the parameters in config file to the main commands
    for i in range(len(commands)):
        commands[i].append(f'--dataset')
        commands[i].append(f'{args.dataset}')
        commands[i].append(f'--lr')
        commands[i].append(f'{args.lr}')
        commands[i].append(f'--use_optimal_lr')
        commands[i].append(f'{args.use_optimal_lr}')
        commands[i].append(f'--prediction_layer')
        commands[i].append(f'{args.prediction_layer}')
        for key, value in train_config.items():
            commands[i].append(f'--{key}')
            commands[i].append(f'{value}')
        commands[i].append(f'--train')
        commands[i].append(f'{args.train}')
        commands[i].append(f'--checkpoint_path')
        commands[i].append(f'{args.checkpoint_path}')
        commands[i].append(f'--original_index')
        commands[i].append(f'{args.original_index}')

    # Call each subprocess
    for command_list in commands:
        subprocess.call(command_list)

if __name__ == '__main__':
    run_egn()