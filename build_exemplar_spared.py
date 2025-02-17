import os
import torch
import heapq
import torchvision
import numpy as np
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from torchvision.transforms import Compose, Normalize
from spared.spared_datasets import get_dataset
from spared.dataloaders import get_pretrain_dataloaders

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

# Add argparse
parser = argparse.ArgumentParser(description="Arguments for training EGGN")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
parser.add_argument("--prediction_layer", type=str, default="c_d_log1p", help="Layer to use for prediction")
parser.add_argument('--batch_size_dataloader', type=int, default=64, help='Batch size')
parser.add_argument('--num_cores', type=int, default=12, help='Number of cores')
parser.add_argument('--numk', type=int, default=6, help='Number of k')
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
parser.add_argument("--max_steps", type=int, default=100, help="Max steps")
parser.add_argument("--val_interval", type=float, default=0.8, help="Validation interval")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument('--use_optimal_lr', type=str2bool, default=False, help='Whether or not to use the optimal learning rate in csv for the dataset.')
parser.add_argument("--verbose_step", type=int, default=10, help="Verbose step")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--mdim", type=int, default=512, help="Dimension of the message")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--optim_metric", type=str, default="MSE", help="Metric to optimize")
parser.add_argument("--patches_key", type=str, default="patches_scale_1.0", help="Key of the patches in the dataset")
parser.add_argument("--graph_radius", type=float, default=1000, help="Graph radius")
parser.add_argument("--train", type=str2bool, default=True, help="Train or load the model")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Get dataset according to the dataset name passed in args
dataset = get_dataset(args.dataset, visualize=False)

# Declare data loaders
train_dl, val_dl, test_dl = get_pretrain_dataloaders(dataset.adata, layer=args.prediction_layer, 
                                                             batch_size = args.batch_size_dataloader,
                                                             shuffle = False,
                                                             use_cuda = use_cuda)

dataloaders = {f"train_{args.dataset}": train_dl, f"val_{args.dataset}": val_dl}
# Add test loader only if it is not None
if test_dl is not None:
    dataloaders[f"test_{args.dataset}"] = test_dl

save_dir = f"exemplars/{args.dataset}"
TORCH_HOME = os.path.join('.torch')
os.makedirs(TORCH_HOME, exist_ok=True)

os.makedirs(os.path.join(save_dir),exist_ok=True)
os.environ['TORCH_HOME'] = TORCH_HOME
#encoder = torchvision.models.resnet18(weights="IMAGENET1K_V1")
encoder = torchvision.models.resnet18(True)
features = encoder.fc.in_features
modules=list(encoder.children())[:-1]
encoder=torch.nn.Sequential(*modules)
for p in encoder.parameters():
    p.requires_grad = False
encoder=encoder.cuda()
encoder.eval()

transforms = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def generate():
                    
    def extract(patches):
        # recibe tensor con los parches ya recuperados en 2D y los pasa por el encoder
        # el tensor viene de dataloader(batch).obsm['patches_scale_1.0'] con shape [batch size(64), 3, 224, 224]
        return encoder(patches).view(-1,features) # shape post encoder [64, 2048, 1, 1] -> pasa a ser [64, 2048]
    
    for save_name, dataloader in dataloaders.items():
        img_embedding = []
        for data in tqdm(dataloader):
            # Get patches of the whole slide image contained in dataloader
            tissue_tiles = data.obsm[args.patches_key]
            tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], round(np.sqrt(tissue_tiles.shape[1]/3)), round(np.sqrt(tissue_tiles.shape[1]/3)), -1))
            # Permute dimensions to be in correct order for normalization
            tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
            # Make transformations in tissue tiles
            tissue_tiles = tissue_tiles/255.
            # Transform tiles
            tissue_tiles = transforms(tissue_tiles)
            # extract patch encoding
            img_embedding += [extract(tissue_tiles)]
        # Embedding of all the patches of the loaded image
        img_embedding = torch.cat(img_embedding).contiguous()
        # Save tensor of shape [#patches, 2048]. This tensor corresponds to the whole image encoded.
        torch.save(img_embedding, f"{save_dir}/{save_name}.pt")

generate()

def create_search_index(save_name):
    
    class Queue:
        def __init__(self, max_size = 2):
            self.max_size = max_size
            self.list = []

        def add(self, item):

            heapq.heappush(self.list, item)

            while len(self.list) > self.max_size:
                heapq.heappop(self.list)
    
        def __repr__(self):
            return str(self.list)
    
    p = torch.load(f"{save_dir}/{save_name}.pt").cuda() 
    Q = [Queue(max_size=128) for _ in range(p.size(0))]   
    op = torch.load(f"{save_dir}/train_{args.dataset}.pt").cuda()
    dist = torch.cdist(p.unsqueeze(0),op.unsqueeze(0),p = 1).squeeze(0)
    
    if "train" in save_name:
         # If we are getting the KNNs of the train dataset, ignore the closest NN (it is the patch itself)
        topk = min(len(dist), 101)
        # Take the top 100 distances and save their value and index
        knn = dist.topk(topk, dim = 1, largest=False) 
        q_values = knn.values[:, 1:].cpu().numpy()
        q_infos =  knn.indices[:, 1:].cpu().numpy() 
    
    else:
        topk = min(len(dist),100)
        knn = dist.topk(topk, dim = 1, largest=False)
        q_values = knn.values.cpu().numpy()
        q_infos =  knn.indices.cpu().numpy()

    def add(q_value,q_info, myQ):
        for idx in range(len(q_value)):
            myQ.add((-q_value[idx],q_info[idx],save_name))
        return myQ

    Q = Parallel(n_jobs=args.num_cores)(delayed(add)(q_values[f],q_infos[f],Q[f]) for f in range(q_values.shape[0]))
    np.save(f"{save_dir}/{save_name}.npy", [myq.list for myq in Q])
    
for save_name in dataloaders.keys():
    create_search_index(save_name)
        
        
                