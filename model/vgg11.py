import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class affordance_model(nn.Module):
    def __init__(self, originalModel):
        super(affordance_model, self).__init__()
        self.features = nn.Sequential(*list(originalModel.features))
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout()
        )
        self.pinch = nn.Linear(4096, 10)
        self.clench = nn.Linear(4096, 10)
        self.poke = nn.Linear(4066, 10)
        self.palm = nn.Linear(4096, 10)
        w = torch.FloatTensor([[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w = w.to(device)

    def weighted_sum(self, x):
        return ((self.w * x).sum(-1, keepdim=True)-1)*100/9

    def forward(self, x, size):
        x = self.features(x)
        x = x.view(-1, 512*7*7)
        
        x = self.classifier(x)

        pinch = F.softmax(self.pinch(x), dim=-1)
        clench = F.softmax(self.clench(x), dim=-1)
        poke = F.softmax(self.poke(x), dim=-1)
        palm = F.softmax(self.palm(x), dim=-1)

        pinch = self.weighted_sum(pinch)
        clench = self.weighted_sum(clench)
        poke = self.weighted_sum(poke)
        palm = self.weighted_sum(palm)

        return pinch, clench, poke, palm