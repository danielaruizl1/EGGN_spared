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
pip install spared

export PYTHONPATH=/anaconda3/envs/eggn_spared/lib/python3.10/site-packages/spared:$PYTHONPATH
```

## Train EGGN

To use a dataset from spared, train eggn by running:
```bash
python run_main_config.py --dataset spared_dataset_name
```

To use a dataset not included in spared, train eggn by running:
```bash
python run_main_config.py --dataset adata_path
```
`adata_path` should be the path to an adata.h5ad


