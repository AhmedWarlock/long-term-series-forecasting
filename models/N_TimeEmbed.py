import torch
import torch.nn as nn
import numpy as np
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
        self.in_channels = args.seq_len + 32
        self.out_channels = args.pred_len
        self.num_feats = args.enc_in
        self.individual = args.individual
        self.hour_embed_layer = nn.Embedding(24,16)
        self.day_embed_layer = nn.Embedding(266,16)

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

            day_last = time_stamp[:,-1,0].detach()
            hour_last = time_stamp[:,-1,1].detach()

            day_embd = self.day_embed_layer(day_last.long())
            hour_embd = self.hour_embed_layer(hour_last.long())

            time_embd = torch.cat((day_embd,hour_embd), dim = 1)
            time_embd = time_embd.unsqueeze(-1).repeat(1,1,in_feats)
            input = torch.cat((x,time_embd), dim=1)
            if self.individual:
                output = self.lin(input)         
            else:
                output = self.lin(input.permute(0,2,1)).permute(0,2,1)
            
            output = output + seq_last
            return output 
    





    # def forward(self, x, time_stamp):
    #         batch_size, _, in_feats = x.shape
    #         time_last = time_stamp[:,-1,:].detach()
    #         temp_enc = self.time_embed_layer(time_last.squeeze(-1).long()) # [16, 1, 48]
    #         # temp_enc =temp_enc.mean(dim=1)
    #         temp_enc =temp_enc.squeeze(1)
    #         temp_enc = temp_enc.unsqueeze(-1).repeat(1,1,in_feats)
    #         # temp_enc = self.temp_enc_layer(time_stamp)
    #         input = torch.cat((x,temp_enc), dim = 1)
    #         # input = x + temp_enc            
    #         if self.individual:
    #             output = self.lin(input)         
    #         else:
    #             output = self.lin(input.permute(0,2,1)).permute(0,2,1)
    #             output = self.proj_layer(output)
    #         return output