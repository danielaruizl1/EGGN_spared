import json
import subprocess
import argparse
import os

def run_egn():
    
    # Get parsed the path of the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='Dataset name to find the config file')
    args = parser.parse_args()

    current_dir = os.path.dirname(__file__)

    commands = [
        ['python', os.path.join(current_dir,'build_exemplar_sepal.py')],
        ['python', os.path.join(current_dir,'generate_graph_sepal.py')],
        ['python', os.path.join(current_dir,'main_sepal.py')]
    ]

    for i in range(len(commands)):
        commands[i].append('--dataset')
        commands[i].append(f'{args.dataset}')

    # Call each subprocess
    for command_list in commands:
        subprocess.call(command_list)

if __name__ == '__main__':
    run_egn()