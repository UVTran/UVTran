import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size,drop=0.1):
        super(ResidualBlock, self).__init__()

        self.input_norm1 = nn.LayerNorm(hidden_size)
        self.prefc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(drop)
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.prefc2 = nn.Linear(hidden_size, input_size)
        self.input_norm2 = nn.LayerNorm(input_size)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):
        out = self.prefc1(x)
        out = self.input_norm1(out)
        out = self.act1 (out)
        out =x+self.dropout1(out)
        x = out
        out = self.prefc2(out)
        out = self.input_norm2(out)
        out = self.act2(out)
        out = x + self.dropout2(out)
        return out

class skipBlock(nn.Module):
    def __init__(self, input_size, hidden_size,drop=0.1):
        super(skipBlock, self).__init__()

        self.input_norm1 = nn.LayerNorm(hidden_size)
        self.prefc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(drop)
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.prefc2 = nn.Linear(hidden_size, input_size)
        self.input_norm2 = nn.LayerNorm(input_size)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):

        out = self.prefc1(x)
        out = self.input_norm1(out)
        out = self.act1 (out)
        out =self.dropout1(out)

        out = self.prefc2(out)
        out = self.input_norm2(out)
        out = self.act2(out)
        out = x + self.dropout2(out)
        return out

class nonemaxModel(torch.nn.Module):

    def __init__(self,num_blocks,drop=0.1,input_size =6,sample_num=297, output_size =200,classnum=100):
        super().__init__()

        super().__init__()
        self.input_size = input_size

        self.hideen_size1 = 16


        self.hideen_size4 = 32
        self.hideen_size5 = 64

        self.hideen_size6 = self.hideen_size5*sample_num

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)

        self.input_norm1 = nn.LayerNorm(self.hideen_size1)

        self.input_norm3 = nn.LayerNorm(self.hideen_size4)
        self.input_norm4 = nn.LayerNorm(self.hideen_size5)

        self.localfc1 = nn.Linear(self.input_size, self.hideen_size1)

        self.globalfc1 = nn.Linear(self.hideen_size1, self.hideen_size4)
        self.globalfc2 = nn.Linear(self.hideen_size4, self.hideen_size5)


        # self.layers = nn.ModuleList()
        #
        # for _ in range(num_blocks):
        #     self.layers.append(ResidualBlock(self.hideen_size6,self.hideen_size6))

        # self.pre_linear1 = nn.Linear(self.hideen_size6,self.hideen_size6)
        # self.preinput_norm1 = nn.LayerNorm(self.hideen_size6)
        # self.dropout1 = nn.Dropout(drop)

        self.pre_linear2 = nn.Linear(self.hideen_size6, output_size*classnum)

    def forward(self, x):

        x = self.localfc1(x)
        x = self.input_norm1(x)
        x = self.act1(x)


        x = self.globalfc1(x)
        x = self.input_norm3(x)
        x = self.act1(x)

        x = self.globalfc2(x)
        x = self.input_norm4(x)
        x = self.act1(x)
        x = x.reshape(x.size(0),-1)
        # x = torch.max(x, 1)[0]


        # for layer in self.layers:
        #     x = layer(x)  # 通过所有的残差层

        # x = self.pre_linear1(x)
        # x = self.preinput_norm1(x)
        # x = self.act1(x)
        # x = self.dropout1(x)
        x = self.pre_linear2(x)  # 最后输出
        return x


class preModel(torch.nn.Module):

    def __init__(self,num_blocks,drop=0.1,input_size =6, output_size =200,classnum=100):
        super().__init__()

        super().__init__()
        self.input_size = input_size

        self.hideen_size1 = 32
        self.hideen_size4 = 128
        self.hideen_size5 = 256
        self.hideen_size6 = self.hideen_size5*2

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.act3 = nn.PReLU(num_parameters=1, init=0.25)

        self.input_norm1 = nn.LayerNorm(self.hideen_size1)

        self.input_norm3 = nn.LayerNorm(self.hideen_size4)
        self.input_norm4 = nn.LayerNorm(self.hideen_size5)

        self.localfc1 = nn.Linear(self.input_size, self.hideen_size1)

        self.globalfc1 = nn.Linear(self.hideen_size1, self.hideen_size4)
        self.globalfc2 = nn.Linear(self.hideen_size4, self.hideen_size5)


        self.layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.layers.append(ResidualBlock(self.hideen_size6,self.hideen_size6))

        # self.pre_linear1 = nn.Linear(self.hideen_size5,self.hideen_size5)
        # self.preinput_norm1 = nn.LayerNorm(self.hideen_size5)
        self.dropout1 = nn.Dropout(drop)

        self.pre_linear2 = nn.Linear(self.hideen_size6, output_size*classnum)

    def forward(self, x):

        x = self.localfc1(x)
        x = self.input_norm1(x)
        x = self.act1(x)


        x = self.globalfc1(x)
        x = self.input_norm3(x)
        x = self.act2(x)

        x = self.globalfc2(x)
        x = self.input_norm4(x)
        x = self.act3(x)

        max_x = torch.max(x, 1)[0]
        min_x = torch.min(x, 1)[0]
        x = torch.cat((max_x,min_x),1)

        for layer in self.layers:
            x = layer(x)  # 通过所有的残差层

        # x = self.pre_linear1(x)
        # x = self.preinput_norm1(x)
        # x = self.act1(x)
        x = self.dropout1(x)
        x = self.pre_linear2(x)  # 最后输出
        return x


class wiseModel(torch.nn.Module):

    def __init__(self,num_blocks,drop=0.1,input_size =6, output_size =200,classnum=100):
        super().__init__()

        super().__init__()
        self.input_size = input_size

        self.hideen_size1 = 256
        self.hideen_size4 = 128*4
        self.hideen_size5 = 512*4
        self.hideen_size6 = self.hideen_size5

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.act3 = nn.PReLU(num_parameters=1, init=0.25)

        self.input_norm1 = nn.LayerNorm(self.hideen_size1)

        self.input_norm3 = nn.LayerNorm(self.hideen_size4)
        self.input_norm4 = nn.LayerNorm(self.hideen_size5)

        self.localfc1 = nn.Linear(self.input_size, self.hideen_size1)

        self.globalfc1 = nn.Linear(self.hideen_size1, self.hideen_size4)
        self.globalfc2 = nn.Linear(self.hideen_size4, self.hideen_size5)


        # self.layers = nn.ModuleList()
        #
        # for _ in range(num_blocks):
        #     self.layers.append(ResidualBlock(self.hideen_size6,self.hideen_size6))

        # self.pre_linear1 = nn.Linear(self.hideen_size5,self.hideen_size5)
        # self.preinput_norm1 = nn.LayerNorm(self.hideen_size5)
        self.dropout1 = nn.Dropout(drop)

        self.pre_linear2 = nn.Linear(self.hideen_size6, output_size*classnum)

    def forward(self, x):

        x = self.localfc1(x)
        x = self.input_norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)


        x = self.globalfc1(x)
        x = self.input_norm3(x)
        x = self.act2(x)
        x = self.dropout1(x)

        x = self.globalfc2(x)
        x = self.input_norm4(x)
        x = self.act3(x)

        x = torch.max(x, 1)[0]
        # min_x = torch.min(x, 1)[0]
        # x = torch.cat((max_x,min_x),1)

        x = self.dropout1(x)
        x = self.pre_linear2(x)  # 最后输出
        return x

class tranModel(torch.nn.Module):

    def __init__(self,num_blocks,drop=0.1,input_size =6, output_size =200,classnum=100):
        super().__init__()

        self.input_size = input_size

        self.hideen_size1 = 32
        self.hideen_size4 = 64
        self.hideen_size5 = 512
        self.hideen_size6 = self.hideen_size5+self.hideen_size4+self.hideen_size1

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.act3 = nn.PReLU(num_parameters=1, init=0.25)

        self.input_norm1 = nn.LayerNorm(self.hideen_size1)

        self.input_norm3 = nn.LayerNorm(self.hideen_size4)
        self.input_norm4 = nn.LayerNorm(self.hideen_size5)

        self.localfc1 = nn.Linear(self.input_size, self.hideen_size1)

        self.globalfc1 = nn.Linear(self.hideen_size1, self.hideen_size4)
        self.globalfc2 = nn.Linear(self.hideen_size4, self.hideen_size5)


        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()

        for _ in range(num_blocks):
            self.layers1.append(ResidualBlock(self.hideen_size1,self.hideen_size1))
            self.layers2.append(ResidualBlock(self.hideen_size4, self.hideen_size4))
            self.layers3.append(ResidualBlock(self.hideen_size5, self.hideen_size5))

        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.dropout1 = nn.Dropout(drop)

        self.knot_linear = nn.Linear(self.hideen_size1, 8)
        self.vo_linear = nn.Linear(self.hideen_size6, output_size*classnum)

    def forward(self, x):

        x = self.localfc1(x)
        x = self.input_norm1(x)
        x = self.act1(x)
        max1 = torch.max(x, 1)[0]


        x = self.globalfc1(x)
        x = self.input_norm3(x)
        x = self.act2(x)
        max2 = torch.max(x, 1)[0]

        x = self.globalfc2(x)
        x = self.input_norm4(x)
        x = self.act3(x)

        max3 = torch.max(x, 1)[0]

        for block in  range(len(self.layers1)):
            max1 = self.layers1[block](max1)
            max2 = self.layers2[block](max2)
            max3 = self.layers3[block](max3)

        max1 = self.dropout1(max1)
        max2 = self.dropout1(max2)
        max3 = self.dropout1(max3)

        knotfea = max1
        x = torch.cat((max1,max2,max3),1)

        knotfea = self.knot_linear(knotfea)
        vofea = self.vo_linear(x)
        return knotfea,vofea


