## Set up

Run the following to define your environment in terminal:

```bash
conda create -n eggn_spared
conda activate eggn_spared
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install torch-geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install squidpy
pip install wandb
pip install wget
pip install combat
pip install opencv-python
pip install positional-encodings[pytorch]
pip install openpyxl
pip install pyzipper
pip install plotly
pip install sh
pip install timm
```

## Dataset

To use any of the 26 datasets within spared, download our preprocessed data by running:
```
wget http://157.253.243.29/SpaRED/processed_data.tar.gz
```

Please structure the data files as follows.
```
└───v2
    └───spared
	└───processed_data
```
## Train EGGN

To use a dataset from spared, train eggn by running:
```bash
cd v2
python run_main_config.py --dataset spared_dataset_name
```

To use a dataset not included in spared, train eggn by running:
```bash
cd v2
python run_main_config.py --dataset adata_path
```
`adata_path` should be the path to an adata.h5ad


