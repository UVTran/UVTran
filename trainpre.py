import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import dataprocess
import pnet
from torch.cuda import amp

scaler = amp.GradScaler()

def get_args_parser():
    parser = argparse.ArgumentParser('bsp-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150000, type=int)

    parser.add_argument('--saveepochs', default=6000, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--in_path', default='D:/data/g1data/input', type=str,
                        help='dataset path')

    parser.add_argument('--out_path', default='D:/data/g1data/output', type=str,
                        help='dataset path')

    parser.add_argument('--vout_path', default='D:/data/g1data/lvoutput', type=str,
                        help='dataset path')

    return parser


def main(args):

    device = torch.device('cuda:0')
    clnum =10
    model = pnet.encoarModel(12, drop=0.05, input_size=3, output_size=192, classnum=clnum)
    model.to(device)
    print(args.batch_size)
    print("line0")
    batch_size = args.batch_size
    lr = args.lr

    criterion = nn.MSELoss()
    criterion1 = nn.CrossEntropyLoss()  # 均方误差损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))

    dataset_train = dataprocess.TrainmeanDataset(args.in_path,args.vout_path)
    print(dataset_train.__len__())
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


    for epoch in range(args.epochs):
        # print("step:", epoch)
        losssumpo = 0
        model.train(True)
        for step, (derivative, vox) in enumerate(data_loader_train):
            # print('step', step)
            derivative = derivative.to(device)
            voxel = vox.long().to(device)
            coarpo = model(derivative)

            # loss1 = criterion(knot,points)
            loss2 = criterion1(coarpo.reshape(-1,clnum),voxel.reshape(-1))
            loss = loss2
            losssumpo +=loss2.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.step()
        if (epoch + 1) % 1 == 0:
            print("epoch:", epoch, 'losssumpo:', losssumpo)
            if (epoch + 1) % 4 == 0:

                if (epoch + 1) % 1000 == 0:
                    lr = lr*0.1
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
                checkpoint_path = 'net'
                checkpoint_path = os.path.join(checkpoint_path, '2d' + str(int((epoch+1)/4)) + '.pt')
                torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

