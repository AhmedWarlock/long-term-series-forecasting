import torch
import torch.nn as nn
import numpy as np
import pdb

class PatchBlock(nn.Module):

    def __init__(self,in_channels, out_channels, num_feats):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.patch_len = 16
        self.patch_dim = 16 * 2
        self.stride = 8
        self.patch_num = (in_channels - self.patch_len)//self.stride + 1
        self.num_feats = num_feats
        self.patch_embed_layer = nn.Linear(self.patch_len, self.patch_dim)
        self.flatten = nn.Flatten(start_dim=-2)
        for _ in range(num_feats) :
                self.fc_layers.append(nn.Linear(self.patch_dim, self.patch_dim))
                self.proj_layers.append(nn.Linear(self.patch_dim*self.patch_num, self.out_channels))

    
    def forward(self, x):
        # patching
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        #patch embedding
        x = self.patch_embed_layer(x)

        output = torch.zeros_like(x).to(x.device)
        output2 = torch.zeros([x.shape[0],x.shape[1],self.out_channels]).to(x.device)
        for i in range(self.num_feats):
                    output[:,i] = self.fc_layers[i](x[:,i])

        output = self.flatten(output) # [batch 16, feats 21, patch num 41 * dim 256 ]

        for i in range(self.num_feats):
                    output2[:,i] = self.proj_layers[i](output[:,i]) #[16, 21, pred]
        return output2



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_channels = args.seq_len
        self.out_channels = args.pred_len
        self.num_feats = args.enc_in
        self.individual = args.individual
        self.num_layers = args.num_lin_layers
        # self.patch_len = 16
        # self.patch_dim = 16 * 16
        # self.stride = 8
        # self.patch_num = (args.seq_len - self.patch_len)//self.stride + 1


        self.network = nn.Sequential(
                PatchBlock(self.in_channels, self.out_channels,self.num_feats),
                *[PatchBlock(self.out_channels, self.out_channels,self.num_feats) for _ in range(self.num_layers-1)]
        )


    def forward(self, x):
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
            
            output = self.network(x.permute(0,2,1)).permute(0,2,1)         
            output = output + seq_last
            return output 