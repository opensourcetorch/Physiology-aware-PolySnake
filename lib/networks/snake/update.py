import torch
import torch.nn as nn
import ipdb


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1))) 

        h = (1-z) * h + z * q
        
        return h
class ConvGRU2d(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64):
        super(ConvGRU2d, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [1,3], padding=[0,1])
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [1,3], padding=[0,1])
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [1,3], padding=[0,1])

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q # h: [batch, 64, 1,128 ]
        # ipdb.set_trace()

        return h


class BasicUpdateBlock(nn.Module):
    def __init__(self):
        super(BasicUpdateBlock, self).__init__()
        self.gru = ConvGRU(hidden_dim=64, input_dim=64)

        self.prediction = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 2, 1)
            )
        # B,2,128

    def forward(self, net, i_poly_fea):
        net = self.gru(net, i_poly_fea)
        offset = self.prediction(net).permute(0, 2, 1) # [batch,128,2]

        return net, offset

class BasicUpdateBlock2d(nn.Module):
    def __init__(self):
        super(BasicUpdateBlock2d, self).__init__()
        self.gru = ConvGRU2d(hidden_dim=64, input_dim=64)

        self.prediction = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 2, 1) # [batch, xy=2, 2, 128]
        )

    def forward(self, net, i_poly_fea):
        net = self.gru(net, i_poly_fea)

        offset = self.prediction(net) # [batch,128,4]
        B,C, _,L = offset.shape
        offset = offset.permute(0,2,3,1) # [batch,2,128,xy=2]
        # ipdb.set_trace()

        return net, offset
