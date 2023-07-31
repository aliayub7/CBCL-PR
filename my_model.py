import torch
import torch.nn as nn
from torch.nn import Linear,ReLU,Dropout
#torch.manual_seed(1)

# define model
class Net(nn.Module):
    def __init__(self,dim,total_classes,seed):
        super(Net,self).__init__()
        torch.manual_seed(seed)
        self.fc_layers = nn.Sequential(
        #Linear(dim,dim),
        #ReLU(True),
        #Dropout(),
        #Linear(dim,dim),
        #ReLU(True),
        #Dropout(),
        Linear(dim,total_classes)
        )
        #self.layer_dictionary = nn.ModuleDict({"softmax":Linear(dim,total_classes)})
    def forward(self,x):
        x = self.fc_layers(x)
        #x = self.layer_dictionary['softmax'](x)
        return x
