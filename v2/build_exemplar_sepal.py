import os
import torch
import heapq
import torchvision
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
from torchvision.transforms import Compose, Normalize

# Current path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# Add path to ST repository
sepal_dir = os.path.join(parent_dir[:-4], 'SEPAL')
sys.path.append(sepal_dir)

# Import SEPAL utils
from utils import *

parser_ST = get_main_parser()
args_ST = parser_ST.parse_args()
use_cuda = torch.cuda.is_available()

# Get dataset from the values defined in args
dataset = get_dataset_from_args(args=args_ST)
# Declare data loaders
train_dl, val_dl, test_dl = dataset.get_pretrain_dataloaders(layer=args_ST.prediction_layer, batch_size = args_ST.batch_size, shuffle = args_ST.shuffle, use_cuda = use_cuda)
dataloaders = {f"train_{args_ST.dataset}": train_dl, f"val_{args_ST.dataset}": val_dl}
# Add test loader only if it is not None
if test_dl is not None:
    dataloaders[f"test_{args_ST.dataset}"] = test_dl

save_dir = f"{current_dir}/exemplars/{args_ST.dataset}"
TORCH_HOME = os.path.join('.torch')
os.makedirs(TORCH_HOME, exist_ok=True)

os.makedirs(os.path.join(save_dir),exist_ok=True)
os.environ['TORCH_HOME'] = TORCH_HOME
encoder = torchvision.models.resnet18(True)
features = encoder.fc.in_features
modules=list(encoder.children())[:-1]
encoder=torch.nn.Sequential(*modules)
for p in encoder.parameters():
    p.requires_grad = False
encoder=encoder.cuda()
encoder.eval()

num_cores = 12
batch_size = 64
window = 256   

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
            tissue_tiles = data.obsm['patches_scale_1.0']
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
        print(img_embedding.size())
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
    op = torch.load(f"{save_dir}/{save_name}.pt").cuda()
    dist = torch.cdist(p.unsqueeze(0),op.unsqueeze(0),p = 1).squeeze(0)
    topk = min(len(dist),100)
    knn = dist.topk(topk, dim = 1, largest=False)

    #Quitar primera posición (sacar 101)
    q_values = knn.values.cpu().numpy()
    q_infos =  knn.indices.cpu().numpy() 

    def add(q_value,q_info, myQ):
        for idx in range(len(q_value)):
            myQ.add((-q_value[idx],q_info[idx],save_name))
        return myQ

    Q = Parallel(n_jobs=num_cores)(delayed(add)(q_values[f],q_infos[f],Q[f]) for f in range(q_values.shape[0]))
    np.save(f"{save_dir}/{save_name}.npy", [myq.list for myq in Q])
    
for save_name in dataloaders.keys():
    create_search_index(save_name)
        
        
                