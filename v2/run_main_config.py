import json
import subprocess
import argparse

def run_egn():
    
    # Auxiliary function to use booleans in parser
    str2bool = lambda x: (str(x).lower() == 'true')
    str2intlist = lambda x: [int(i) for i in x.split(',')]
    str2floatlist = lambda x: [float(i) for i in x.split(',')]
    str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

    # Get parsed the path of the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="10xgenomic_human_breast_cancer", help='Dataset name')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--use_optimal_lr', type=str2bool, default=False, help='Whether or not to use the optimal learning rate in csv for the dataset.')
    parser.add_argument('--prediction_layer', type=str, default="c_d_log1p", help='Layer to use for prediction')
    parser.add_argument('--train_config', type=str, default="train_EGGN_config.json", help='Config file path with train hyperparameters')
    args = parser.parse_args()

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

    # Call each subprocess
    for command_list in commands:
        subprocess.call(command_list)

if __name__ == '__main__':
    run_egn()