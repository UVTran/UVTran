import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import dataprocess
import pnet
import torch.nn.functional as F
from collections import OrderedDict
import torch.distributed as dist
import torch.multiprocessing as mp


def get_args_parser():
    parser = argparse.ArgumentParser('bsp-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150000, type=int)

    parser.add_argument('--saveepochs', default=6000, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--in_path', default='lessdata/input', type=str,
                        help='dataset path')

    parser.add_argument('--out_path', default='lessdata/output', type=str,
                        help='dataset path')

    parser.add_argument('--vout_path', default='lessdata/voutput', type=str,
                        help='dataset path')

    return parser


def main(args):
    device_ids =[1]
    device = torch.device('cuda:0')
    clnum =100
    model = pnet.tranModel(12, drop=0.2,input_size=3,output_size=192,classnum=clnum)

    # 通过 DataParallel 包装模型
    # print(torch.cuda.device_count())
    # checkpoint = torch.load('net/layer1003d31.pt')
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint.items():
    #     if k.startswith('module.'):
    #         new_state_dict[k[7:]] = v  # 去掉前缀
    #     else:
    #         new_state_dict[k] = v
    #
    #         # 4. 加载新的状态字典
    # model.load_state_dict(new_state_dict)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids=device_ids)
    # model.load_state_dict(torch.load('net/tranvo6.pt'))


        # 将模型移动到 GPU
    model.to(device)
    print(args.batch_size)
    print("line0")
    batch_size = args.batch_size
    lr = args.lr

    criterion = nn.MSELoss()
    criterion1 = nn.CrossEntropyLoss()  # 均方误差损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))

    dataset_train = dataprocess.TrainvoxelDataset(args.in_path, args.out_path,args.vout_path)
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
        for step, (derivative, points,vox) in enumerate(data_loader_train):
            print('step', step)
            derivative = derivative[:,:,:3].to(device)
            points = points[:,:8].to(device)
            voxel = vox.long().to(device)
            knot,coarpo = model(derivative)


            loss1 = criterion(knot,points)
            loss2 = criterion1(coarpo.reshape(-1, clnum),voxel.reshape(-1))
            loss = loss1+loss2
            losssumpo +=loss2.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        if (epoch + 1) % 5 == 0:
            print("epoch:", epoch, 'losssumpo:', losssumpo)
            if (epoch + 1) % 20 == 0:

                if (epoch + 1) % 1000 == 0:
                    lr = lr*0.1
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
                checkpoint_path = 'net'
                checkpoint_path = os.path.join(checkpoint_path, 'tranvo' + str(int(epoch/1)) + '.pt')
                torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

