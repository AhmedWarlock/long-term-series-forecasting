import torch
import torch.nn as nn
from sklego.preprocessing import RepeatingBasisFunction
import pdb

class LinearLayer(nn.Module):

    def __init__(self,in_channels, out_channels, num_feats):
        super().__init__()
        self.layers = nn.ModuleList()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_feats = num_feats
        for _ in range(num_feats) :
                self.layers.append(nn.Linear(self.in_channels, self.out_channels))

    
    def forward(self, x):
        output = torch.zeros([x.shape[0],self.out_channels,x.shape[2]]).to(x.device)
        for i in range(self.num_feats):
                    output[:,:,i] = self.layers[i](x[:,:,i])
        
        return output



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_channels = args.seq_len + 16
        self.out_channels = args.pred_len
        self.num_feats = args.enc_in
        self.individual = args.individual
        self.rbf = RepeatingBasisFunction(n_periods=16,input_range=(0,23),
                         	remainder="drop")

        if self.individual:
            self.lin = nn.Sequential(
                  LinearLayer(self.in_channels, self.out_channels, self.num_feats),
                  *[LinearLayer(self.out_channels, self.out_channels,self.num_feats) for _ in range(2)]
            )
        else : 
            self.lin =  nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x, time_stamp):
            _, _, in_feats = x.shape
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last

            hour_last = time_stamp[:,-1].detach() #[16]
            hour_embd = torch.tensor(self.rbf.fit_transform(hour_last.cpu().numpy())
                                    ).float().to(x.device)
            time_embd = hour_embd.unsqueeze(-1).repeat(1,1,in_feats)


            input = torch.cat((x,time_embd), dim=1)


            if self.individual:
                output = self.lin(input)         
            else:
                output = self.lin(input.permute(0,2,1)).permute(0,2,1)
            
            output = output + seq_last
            return output 
