import torch
import torch.nn as nn
import torch.nn.functional as F
import copy as cp

class affordance_model_size_as_input(nn.Module):
    def __init__(self, originalModel):
        super(affordance_model, self).__init__()
        modules = list(originalModel.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.features2 = cp.deepcopy(self.features)
        hidden = 4096
        cnn_out_dim = 2*2048+3
        self.out_dim = 10
        self.pinch = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        self.clench = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        self.poke = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        self.palm = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        w = torch.FloatTensor([list(range(self.out_dim))])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w = w.to(device)

    def weighted_sum(self, x):
        return ((self.w * x).sum(-1, keepdim=True)-1)*100/(self.out_dim-1)

    def forward(self, x, short, long, pixels):
        x1 = self.features(x)
        x2 = self.features2(x)
        x1 = x1.view(-1, 2048)
        x2 = x2.view(-1, 2048)
        short = short.view(-1, 1)
        long = long.view(-1, 1)
        pixels = pixels.view(-1, 1)

        x = torch.cat((x1, x2, short, long, pixels), 1)

        pinch = F.softmax(self.pinch(x), dim=-1)
        clench = F.softmax(self.clench(x), dim=-1)
        poke = F.softmax(self.poke(x), dim=-1)
        palm = F.softmax(self.palm(x), dim=-1)

        pinch = self.weighted_sum(pinch)
        clench = self.weighted_sum(clench)
        poke = self.weighted_sum(poke)
        palm = self.weighted_sum(palm)

        return pinch, clench, poke, palm


class affordance_model(nn.Module):
    def __init__(self, originalModel):
        super(affordance_model, self).__init__()
        modules = list(originalModel.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.features2 = cp.deepcopy(self.features)
        hidden = 4096
        cnn_out_dim = 2*2048
        self.out_dim = 10
        self.pinch = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        self.clench = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        self.poke = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        self.palm = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, self.out_dim)
        )
        self.size = nn.Sequential(
            nn.Linear(cnn_out_dim, 3)
        )
        w = torch.FloatTensor([list(range(self.out_dim))])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w = w.to(device)

    def weighted_sum(self, x):
        return ((self.w * x).sum(-1, keepdim=True)-1)*100/(self.out_dim-1)

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.features2(x)
        x1 = x1.view(-1, 2048)
        x2 = x2.view(-1, 2048)

        x = torch.cat((x1, x2), 1)

        pinch = F.softmax(self.pinch(x), dim=-1)
        clench = F.softmax(self.clench(x), dim=-1)
        poke = F.softmax(self.poke(x), dim=-1)
        palm = F.softmax(self.palm(x), dim=-1)
        size = self.size(x)

        pinch = self.weighted_sum(pinch)
        clench = self.weighted_sum(clench)
        poke = self.weighted_sum(poke)
        palm = self.weighted_sum(palm)

        return pinch, clench, poke, palm, size

