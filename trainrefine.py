import argparse
import os
import torch
from torch import nn
import dataprocess
import pnet
import potran
import trannet
import torch.nn.functional as F
from collections import OrderedDict


def get_args_parser():
    parser = argparse.ArgumentParser('bsp-training', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150000, type=int)

    parser.add_argument('--saveepochs', default=6000, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--in_path', default='D:/data/ig1data/single//input', type=str,
                        help='dataset path')

    parser.add_argument('--out_path', default='D:/data/ig1data/single/moutput', type=str,
                        help='dataset path')

    parser.add_argument('--vout_path', default='D:/data/ig1data/single/voutput', type=str,
                        help='dataset path')

    return parser


def main(args):

    device = torch.device('cuda:0')
    clnum =100
    model = trannet.TransformerModelvo(input_size=3, output_size=3, d_model=512, nhead=1, num_layers=12, dim_feedforward=512,num_blocks=12, dropout=0.01,device=device)
    coamodel = pnet.encoarModel(12, drop=0.2, input_size=3, output_size=192, classnum=clnum//10)


    model.to(device)
    print(args.batch_size)
    print("vo0")
    batch_size = args.batch_size
    lr = args.lr

    # criterion2 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))

    dataset_train = dataprocess.TrainmeanDataset(args.in_path, args.vout_path)
    print(dataset_train.__len__())
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    coamodel.to(device)
    coamodel.eval()

    for epoch in range(args.epochs):
        # print("step:", epoch)
        losssumpo = 0
        model.train(True)
        for step, (derivative, voxel) in enumerate(data_loader_train):

            derivative = derivative[:, :, :3].to(device)

            voxel = voxel.to(device)

            with torch.inference_mode():
                coarpo,deriEmd = coamodel(derivative)

            coarpo = coarpo.reshape(args.batch_size,192, clnum//10)
            coarpo = F.softmax(coarpo,2)
            coarpov = torch.argmax(coarpo, 2, keepdim=False)
            coarpov = coarpov*0.1+0.05
            coarpov = coarpov.reshape(args.batch_size,64, 3)
            #
            #
            poEmd = coamodel.pre(coarpov)

            coarpo = model(poEmd, deriEmd)

            loss2 = criterion2(coarpo.reshape(-1, 100),voxel.reshape(-1))
            losssumpo += loss2.item()
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            # scheduler.step()
        if (epoch + 1) % 1 == 0:
            print("epoch:", epoch, 'losssumpo:', losssumpo)
            if (epoch + 1) % 5 == 0:
                if (epoch + 1) % 1000 == 0:
                    lr = lr * 0.1
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
                checkpoint_path = 'net'
                checkpoint_path = os.path.join(checkpoint_path, 'respre' + str(int(epoch / 5)) + '.pt')
                torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

