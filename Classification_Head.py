import torch
import torch.nn as nn

class Classification_Head(nn.Module):

    def __init__(self,input_dimensions, num_classes=31):
        super(Classification_Head, self).__init__()
        self.input_dimensions=input_dimensions
        self.fc=nn.Linear(input_dimensions,num_classes)

    def forward(self,x):
        ans=self.fc(x)
        return ans
    
    def new(self):
        model_new=Classification_Head(self.input_dimensions)
        return model_new
