import torch
import torch.nn as nn
import torch_geometric.nn as pyg
from geb import EGBBlock
from csra import CSRA
from heteroconv import HeteroConv
from torchvision.transforms import Normalize

'''
code is based on https://pytorch-geometric.readthedocs.io/en/latest/

'''

class HeteroGNN(torch.nn.Module):
    def __init__(self, num_layers = 4, mdim=512, num_genes=250):
        super().__init__()

        hidden_channels = mdim
        out_channels = num_genes
        input_channel = mdim

        self.num_genes = num_genes
        
        self.pretransform_win = pyg.Linear(input_channel,hidden_channels,bias=False)
        self.pretransform_exp = pyg.Linear(input_channel+num_genes,hidden_channels,bias=False)
        self.post_transform = nn.Sequential(
            nn.LeakyReLU(0.2,True),
            pyg.Linear(hidden_channels,hidden_channels,bias=False),
            nn.LeakyReLU(0.2,True),
            )
        self.pretransform_ey = pyg.Linear(num_genes,hidden_channels,bias=False)
        self.leaklyrelu = nn.LeakyReLU(0.2)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('window', 'near', 'window'): pyg.SAGEConv(hidden_channels,hidden_channels),
                ('example', 'close', 'example'): pyg.SAGEConv(hidden_channels,hidden_channels), 
                ('example', 'refer', 'window'): EGBBlock((hidden_channels, hidden_channels,hidden_channels), hidden_channels, hidden_channels, add_self_loops = False), 
            }, aggr='mean')
            self.convs.append(conv)

        self.pool = CSRA(hidden_channels)
        self.lin = pyg.Linear(hidden_channels, out_channels)
        self.test_transforms = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    def forward(self, x_dict, edge_index_dict):
        
        x_dict["example"]  = self.post_transform(self.pretransform_exp(x_dict["example"]))
        x_dict['window'] = self.post_transform(self.pretransform_win(x_dict['window']))
        x_dict["example_y"] = self.pretransform_ey(x_dict["example"][:,-self.num_genes:])
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.leaklyrelu(x) for key, x in x_dict.items()}
        return self.lin(self.pool(x_dict, edge_index_dict))
 