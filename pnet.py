import torch
import torch.nn as nn
import torch.nn.functional as F


class encoarModel(torch.nn.Module):

    def __init__(self, num_blocks, drop=0.1, input_size=6, output_size=200, classnum=100):
        super().__init__()

        self.input_size = input_size

        self.hideen_size1 = 32
        self.hideen_size4 = 128
        self.hideen_size5 = 512

        self.act1 = nn.ReLU()

        self.input_norm1 = nn.LayerNorm(self.hideen_size1)

        self.input_norm2 = nn.LayerNorm(self.hideen_size4)
        self.input_norm3 = nn.LayerNorm(self.hideen_size5)

        self.localfc1 = nn.Linear(self.input_size, self.hideen_size1)

        self.globalfc1 = nn.Linear(self.hideen_size1, self.hideen_size4)
        self.globalfc2 = nn.Linear(self.hideen_size4, self.hideen_size5)

        self.votrapos = nn.Linear(297, 64)
        # self.kntrapos = nn.Linear(297,8)

        # self.positional_encoding = PositionalEncoding(self.hideen_size5)
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hideen_size5, 1, self.hideen_size5, drop, batch_first=True,
                                       activation='gelu'),
            num_layers=num_blocks,
        )
        # self.encoder2 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(self.hideen_size5, 1, self.hideen_size5, drop,batch_first=True,activation='gelu'),
        #     num_layers=num_blocks//4,
        # )
        #
        # self.kn_norm = nn.LayerNorm(self.hideen_size5)
        # self.kn_linear = nn.Linear(self.hideen_size5, 1)

        self.dropout1 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(self.hideen_size5)

        self.vo_linear = nn.Linear(self.hideen_size5, 3 * classnum)

    def forward(self, x, dx):
        x = self.localfc1(x)
        x = self.input_norm1(x)
        x = self.act1(x)

        x = self.globalfc1(x)
        x = self.input_norm2(x)
        x = self.act1(x)

        x = self.globalfc2(x)
        x = self.input_norm3(x)
        x = self.act1(x)

        dc = self.localfc1(dx)
        dc = self.input_norm1(dc)
        dc = self.act1(dc)

        dc = self.globalfc1(dc)
        dc = self.input_norm2(dc)
        dc = self.act1(dc)

        dc = self.globalfc2(dc)
        dc = self.input_norm3(dc)
        dc = self.act1(dc)

        vx = x

        lx = x.transpose(1, 2)
        vox = self.votrapos(lx)
        vox = vox.transpose(1, 2)

        vox = self.encoder1(vox)
        vox = self.dropout1(vox)
        #
        # knx = self.encoder2(knx)
        # knx = self.dropout1(knx)

        vox = self.norm(vox)
        vox = self.vo_linear(vox)

        # knx = self.kn_norm(knx)
        # knx = self.kn_linear(knx)
        # knx = knx.squeeze(-1)
        return vox, vx

    def pre(self, tgt):
        tgt = self.localfc1(tgt)
        tgt = self.input_norm1(tgt)
        tgt = self.act1(tgt)

        tgt = self.globalfc1(tgt)
        tgt = self.input_norm2(tgt)
        tgt = self.act1(tgt)

        tgt = self.globalfc2(tgt)
        tgt = self.input_norm3(tgt)
        tgt = self.act1(tgt)

        return tgt



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
        # self.layers = nn.ModuleList()
        #
        # for _ in range(num_layers):
        #     self.layers.append(ResidualBlock(d_model*2,d_model*2))

        # self.positional_encoding = PositionalEncoding(d_model, dropout=0)


        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_size*100)


    def forward(self, tgt_emb,src_emb):

        # scores = torch.matmul(tgt, src.transpose(1, 2)) / torch.sqrt(torch.tensor(3, dtype=torch.float32))
        #
        # # 计算注意力权重（softmax归一化）
        # attention_weights = F.softmax(scores, dim=-1)  # [B, Q, K]
        #
        # # 加权求和得到输出: [B, Q, K] × [B, K, V] → [B, Q, V]
        # decoded_tgt = torch.matmul(attention_weights, src_emb)
        decoded_tgt = self.decoder(tgt_emb, src_emb)  # (tgt_len, batch_size, d_model)
        #
        # decoded_tgt = tgt_emb -decoded_tgt

        decoded_tgt = self.encoder(decoded_tgt)
        # x = torch.cat((tgt_emb, decoded_tgt), 2)
        # for layer in self.layers:
        #     x = layer(x)
        decoded_tgt = self.dropout(decoded_tgt)

        output = self.fc_out(decoded_tgt)  # (tgt_len, batch_size, output_dim)
        return output


    def predict(self,src,tgt):
        src_emb = self.embedding(src)  # (batch_size, src_len, d_model)
        otgt = tgt
        coarpov = tgt
        for i in range(64):
            tgt = otgt
            tgt_emb = self.embedding(tgt)
            tgt_emb = self.positional_encoding(tgt_emb)
            decoded_tgt = self.decoder(tgt_emb, src_emb)
            output = self.fc_out(decoded_tgt)
            output = output[:,-1,:]
            coarpo = output.reshape(3,100)
            coarpo = F.softmax(coarpo,dim=-1)
            coarpov = torch.argmax(coarpo, 1, keepdim=False)
            coarpov = coarpov*0.01+0.005
            coarpov =  coarpov.unsqueeze(0)
            coarpov = coarpov.unsqueeze(0)
            otgt = torch.cat((otgt,coarpov),1)
        output = otgt[:,1:,:]
        return output


