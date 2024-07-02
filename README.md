# [Spatial Transcriptomics Analysis of Gene Expression Prediction using Exemplar Guided Graph Neural Network](./doc/eggn.pdf)
![](doc/coverpage2.jpg)

## Dependency
* python 3.9.4
* pytorch_lightning 1.5.7
* numpy 1.23.3
* Pandas 1.5.2
* tifffile 2021.7.2
* Pillow 9.3.0
* scanpy 1.9.1
* torch 1.9.1+cu111
* torchvision 0.10.1+cu111
* torch-geometric  2.1.0.post1

## Set up

Run the following to define your environment in terminal:

```bash
conda create -n eggn
conda activate eggn
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch_lightning==1.5.7
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install -U torch torchaudio --no-cache-dir
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install timm
```

## Dataset

Please structure the data files as follows.
```
└───spared
    └───processed_data
	└───author
	    └───dataset
	        adata.h5ad
```

## Train EGGN

```bash
cd v2
python run_main_config.py
```

