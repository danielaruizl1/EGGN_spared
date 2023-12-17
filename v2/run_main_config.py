import json
import subprocess
import argparse

def run_egn(code):
    if code=="exemplars" or code=="graphs":

        # Get parsed the path of the config file
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_config', type=str, default='config_visium.json',    help='Path to the .json file with the configs for stnet.')
        args = parser.parse_args()

        # Read the configs
        with open(args.dataset_config, 'rb') as f:
            config_params = json.load(f)

        # Create the command to run. If sota key is "None" call main.py else call main_sota.py
        if code=="exemplars":
            command_list = ['python', 'build_exemplar_copy.py']
        elif code=="graphs":
            command_list = ['python', 'generate_graph_copy.py']

    elif code=="train":

        with open('train_configEGN.json', 'rb') as f:
            config_params = json.load(f)

        # Create the command to run. If sota key is "None" call main.py else call main_sota.py
        command_list = ['python', 'main_copy.py']

    for key, val in config_params.items():
        command_list.append(f'--{key}')
        command_list.append(f'{val}')

    # Call subprocess
    subprocess.call(command_list)

run_egn("graphs")