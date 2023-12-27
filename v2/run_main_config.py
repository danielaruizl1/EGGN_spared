import json
import subprocess
import argparse
import os

def run_egn():
    
    # Get parsed the path of the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,type=str, help='Dataset name')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--train_config', type=str, default="train_EGGN_config.json", help='Config file path with train hyperparameters')
    args = parser.parse_args()

    commands = [
        ['python', 'build_exemplar_sepal.py'],
        ['python', 'generate_graph_sepal.py'],
        ['python', 'main_EGGN_sepal.py']
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
        for key, value in train_config.items():
            commands[i].append(f'--{key}')
            commands[i].append(f'{value}')

    # Call each subprocess
    for command_list in commands:
        subprocess.call(command_list)

if __name__ == '__main__':
    run_egn()