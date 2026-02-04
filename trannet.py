import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModelvo(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dim_feedforward,num_blocks, dropout=0.0,device='cuda'):
        super(TransformerModelvo, self).__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True),
            num_layers=1
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True,activation='gelu'),
            num_layers=num_layers,
        )



        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_size*100)


    def forward(self, tgt_emb,src_emb):

        decoded_tgt = self.decoder(tgt_emb, src_emb)  # (tgt_len, batch_size, d_model)


        decoded_tgt = self.encoder(decoded_tgt)
        decoded_tgt = self.dropout(decoded_tgt)

        output = self.fc_out(decoded_tgt)  # (tgt_len, batch_size, output_dim)
        return output
