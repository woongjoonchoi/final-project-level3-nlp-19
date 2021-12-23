import torch
import torch.nn as nn

import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

import transformers
import torch.nn as nn

class MyEfficientNet(nn.Module) :
    

    def __init__(self , num_classes : int = 18):
        super(MyEfficientNet,self).__init__()
        self.EFF = EfficientNet.from_pretrained('efficientnet-b4' , 
                    in_channels =3 ,num_classes = num_classes)
    

    def forward(self , x : torch.Tensor) ->torch.Tensor :
        x = self.EFF(x)
        x= F.softmax(x , dim =1)
        return x



