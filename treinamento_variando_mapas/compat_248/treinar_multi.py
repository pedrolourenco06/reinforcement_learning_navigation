"""Somente definições do modelo anterior, para avaliação compatível."""
import torch
from torch import nn
from ambiente_multi import OBS_DIM, ACOES

class Rede(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(OBS_DIM,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.value=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        self.advantage=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,len(ACOES)))
    def forward(self,x):
        h=self.features(x); a=self.advantage(h)
        return self.value(h)+a-a.mean(1,keepdim=True)

def device_for(name):
    if name=='cpu': return torch.device('cpu')
    if name in ('auto','cuda') and torch.cuda.is_available(): return torch.device('cuda')
    if name=='cuda': raise RuntimeError('CUDA indisponível')
    if name in ('auto','directml'):
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            if name=='directml': raise
    return torch.device('cpu')

@torch.no_grad()
def action(net,state,device):
    return int(net(torch.as_tensor(state,device=device).unsqueeze(0)).argmax(1).item())
