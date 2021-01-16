import torch
import torch.nn.functional as F


class MishCuda(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def foward(self,x):
        return x * torch.tanh(F.softplus(x))