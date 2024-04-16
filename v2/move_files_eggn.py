import shutil
import os
import pandas as pd

# Upload the csv file with the optimal models
df = pd.read_csv('optimal_models_names_ctlog1p.csv')

# Define datafrae of optimal models for specific sota
sota = "eggn"
df_seleccionado = df[['Dataset', sota]]

assert sota in df.columns, AssertionError('Sota still not in csv file')

for index, row in df_seleccionado.iterrows():
    
    # Get path and dataset
    name = row[sota]
    folder_path = os.path.join('checkpoints', name)
    
    if os.path.exists(folder_path):
        dataset = row['Dataset']

        # Creae destination folder
        os.makedirs(os.path.join('optimal_models', sota, dataset), exist_ok=True)
        destination = os.path.join('optimal_models', sota, dataset, name)
        shutil.move(folder_path, destination)

    else:
        print(f'{folder_path} folder not found')