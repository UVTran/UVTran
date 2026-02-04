import argparse
import os
import torch
from torch import nn
import dataprocess
import pnet
import potran
import trannet
import torch.optim as optim
import torch.nn.functional as F
from geomdl import BSpline
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def val(models,coamodel, train_loader, device):
    models.eval()
    coamodel.eval()
    total_loss = 0
    numclass = 100
    for batch_idx, (point, voxel) in enumerate(train_loader):
        if batch_idx < 1000:
            point, voxel = point.to(device),  voxel.to(device)
            with torch.inference_mode():
                coarpo,deriEmd = coamodel(point)

            coarpo = coarpo.reshape(1,192, numclass//10)
            coarpo = F.softmax(coarpo,2)
            coarpov = torch.argmax(coarpo, 2, keepdim=False)
            coarpov = coarpov*0.1+0.05
            coarpov = coarpov.reshape(1,64, 3)

            poEmd = coamodel.pre(coarpov)
            logits = models(poEmd,deriEmd)

            coarpo = logits.reshape(192, numclass)
            coarpo = F.softmax(coarpo, 1)
            coarpov = torch.argmax(coarpo, 1, keepdim=False)

            point = point.reshape(-1, 3)
            revo = 1 / numclass
            inter = revo / 2
            x = coarpov * revo + inter
            x = x.reshape(64, 3)

            srf = BSpline.Surface()
            srf.degree_u = 3
            srf.degree_v = 3
            srf.ctrlpts_size_u = 8
            srf.ctrlpts_size_v = 8
            new_po = x.cpu().detach().numpy()

            po = new_po.tolist()


            knotu = [0, 0, 0, 0]
            step = 1.0 / 5.0
            for i in range(4):
                knotu.append((i + 1) * step)
            for i in range(4):
                knotu.append(1.0)
            srf.knotvector_u = knotu
            srf.knotvector_v = knotu
            srf.ctrlpts = po
            prepo = []
            for uid in range(101):
                pu = uid * 0.01
                for vid in range(101):
                    pv = vid * 0.01
                    newpa = srf.evaluate_single((pu, pv))
                    prepo.append(newpa)
            newx = torch.tensor(prepo).to(device)

            A_sq = (point ** 2).sum(dim=1).unsqueeze(1)  # (300, 1)
            B_sq = (newx ** 2).sum(dim=1).unsqueeze(0)  # (1, 40000)
            cross = point @ newx.t()  # (300, 40000)
            dist_sq = A_sq + B_sq - 2.0 * cross  # (300, 40000)
            dist_sq.clamp_min_(0)
            min_dist_sq, min_indices = dist_sq.min(dim=1)  # (300,), (300,)
            min_dist = torch.sqrt(min_dist_sq)
            min_dist = min_dist.mean()
            total_loss += min_dist.mean().item()
        else:
            break
    print("Total loss: {}".format(total_loss))

def get_args_parser():
    parser = argparse.ArgumentParser('bsp-training', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150000, type=int)

    parser.add_argument('--saveepochs', default=6000, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--in_path', default='D:/data/g1data/input', type=str,
                        help='dataset path')

    parser.add_argument('--out_path', default='D:/data/g1data/output', type=str,
                        help='dataset path')

    parser.add_argument('--vout_path', default='D:/data/g1data/voutput', type=str,
                        help='dataset path')

    return parser


def main(args):

    device = torch.device('cuda:0')
    clnum =100
    model = trannet.TransformerModelvo(input_size=3, output_size=3, d_model=512, nhead=1, num_layers=12, dim_feedforward=512,num_blocks=12, dropout=0.05,device=device)
    coamodel = pnet.encoarModel(8, drop=0.05, input_size=3, output_size=3, classnum=clnum//10)
    # model.load_state_dict(torch.load('net/catpre0.pt',map_location=device))
    # checkpoint = torch.load('net/108.pt',map_location=device)
    # coamodel.load_state_dict(checkpoint)
    # model.load_state_dict(torch.load('net/respre1.pt',map_location=device))

    model.to(device)


    dataset_test = dataprocess.TrainmeanDataset("D:/data/g1data/test/input","D:/data/g1data/test/output")
    print(dataset_test.__len__())
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    coamodel.to(device)
    coamodel.eval()


    val(model,coamodel, data_loader_test, device)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

