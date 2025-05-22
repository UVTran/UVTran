import numpy as np
from geomdl import BSpline
import torch

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def get_voxel_center(voxel_index):

    index_x = voxel_index[0]
    index_y = voxel_index[1]
    index_z = voxel_index[2]

    # 计算体素中心坐标
    center_x = (index_x + 0.5) / 10
    center_y = (index_y + 0.5) / 10
    center_z = (index_z + 0.5) / 10

    center = [center_x, center_y, center_z]
    return center





def shv():
    srf = BSpline.Surface()
    srf.degree_u = 3
    srf.degree_v = 3
    srf.ctrlpts_size_u = 8
    srf.ctrlpts_size_v = 8

    oldsrf = BSpline.Surface()
    oldsrf.degree_u = 3
    oldsrf.degree_v = 3
    oldsrf.ctrlpts_size_u = 8
    oldsrf.ctrlpts_size_v = 8



    fd = torch.load('newpo.pth')

    fd = fd.reshape(-1)
    fd = fd*0.01+0.005

    f = fd.reshape(64, 3)
    po = f.cpu().detach().numpy()



    oldp = torch.load('oldpo.pth')
    oldp = oldp.reshape(-1)
    # oldp = oldp*0.01+0.005
    oldp = oldp.reshape(64, 3)
    oldpo = oldp.cpu().detach().numpy()


    po = po.tolist()
    oldpo =oldpo.tolist()
    cur = torch.load("der.pth").reshape(-1)
    poin = 6
    curx = cur[::poin]
    cury = cur[1::poin]
    curz = cur[2::poin]


    curx = curx.cpu().detach().numpy().tolist()
    cury = cury.cpu().detach().numpy().tolist()
    curz = curz.cpu().detach().numpy().tolist()

    knotu = [0, 0, 0, 0]
    knotv = [0, 0, 0, 0]
    knt = torch.load('knt.pth').reshape(-1)
    knt = knt.cpu().detach().numpy().tolist()

    for i in range(4):
        knotu.append(knt[i])
        knotv.append(knt[i + 4])

    for i in range(4):
        knotu.append(1)
        knotv.append(1)

    srf.knotvector_u = knotu
    srf.knotvector_v = knotv

    srf.ctrlpts = po

    oldsrf.knotvector_u = knotu
    oldsrf.knotvector_v = knotv

    oldsrf.ctrlpts = oldpo

    oldx = np.zeros((101, 101), dtype=float)
    oldy = np.zeros((101, 101), dtype=float)
    oldz = np.zeros((101, 101), dtype=float)

    newx = np.zeros((101, 101), dtype=float)
    newy = np.zeros((101, 101), dtype=float)
    newz = np.zeros((101, 101), dtype=float)

    for uid in range(101):
        pu = uid * 0.01
        for vid in range(101):
            pv = vid * 0.01

            newpa = srf.evaluate_single((pu, pv))
            newx[uid][vid] = newpa[0]
            newy[uid][vid] = newpa[1]
            newz[uid][vid] = newpa[2]

            oldpa = oldsrf.evaluate_single((pu, pv))
            oldx[uid][vid] = oldpa[0]
            oldy[uid][vid] = oldpa[1]
            oldz[uid][vid] = oldpa[2]

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.plot_surface(newx, newy, newz,  cmap='viridis',alpha=0.8, label="predict surface")


    ax1.plot(curx, cury, curz, label="boundary curve", color='red')

    ax1.axis('off')
    ax1.set_xticklabels([])  # 去除X轴刻度标签
    ax1.set_yticklabels([])  # 去除Y轴刻度标签
    ax1.set_zticklabels([])  # 去除Z轴刻度标签

    ax1.set_title('Classification Result')
    ax1.legend()
    plt.tight_layout()

    plt.show()

