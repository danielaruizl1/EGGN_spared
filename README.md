# [Spatial Transcriptomics Analysis of Gene Expression Prediction using Exemplar Guided Graph Neural Network](./doc/eggn.pdf)
![](doc/coverpage2.jpg)

## Set up

Run the following to define your environment in terminal:

```bash
conda create -n eggn python=3.9.4
conda activate eggn
conda install pytorch-lightning=1.5.7 numpy=1.23.3 pandas=1.5.2 tifffile=2021.7.2 pillow=9.3.0 scanpy=1.9.1 -c conda-forge

conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install pytorch_lightning==1.5.7
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install -U torch torchaudio --no-cache-dir
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
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


