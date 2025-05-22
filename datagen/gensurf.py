import math
import numpy as np
from geomdl import BSpline
import os
import random
import torch

import matplotlib

matplotlib.use('TkAgg')

def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)  # 确保轴是单位向量
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)

    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

def normalize(v):
    return v / np.linalg.norm(v)


def rotate_point(point, axis, theta):
    # 使用Rodrigues' 旋转公式
    axis = normalize(axis)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cross_product = np.cross(axis, point)
    dot_product = np.dot(axis, point)

    rotated_point = (point * cos_theta +
                     cross_product * sin_theta +
                     axis * dot_product * (1 - cos_theta))
    return rotated_point

def get_voxel_index(point):

    x = point[0]
    y = point[1]
    z = point[2]
    # 计算每个维度的体素索引
    index_x = min(int(x * 100), 99)
    index_y = min(int(y * 100), 99)
    index_z = min(int(z * 100), 99)

    # 计算体素的唯一标识
    voxel_index = [index_x, index_y, index_z]
    return voxel_index

def genpoints(po):
    rpo = np.array(po)

    noise_std = 0.2
    noise = np.random.normal(0, noise_std, (4, 1))
    rs = random.randint(0, 1)
    rty = random.randint(0, 5)
    if rty == 0:
        no1 = [noise[0][0], noise[0][0], noise[0][0]]
        if rs == 0:
            for j in range(8):
                rpo[0 * 8 + j] = rpo[0 * 8 + j] + no1
                rpo[1 * 8 + j] = rpo[1 * 8 + j] + no1
        else:
            for j in range(8):
                rpo[6 * 8 + j] = rpo[6 * 8 + j] + no1
                rpo[7 * 8 + j] = rpo[7 * 8 + j] + no1
    elif rty == 1:
        no1 = [noise[0][0], noise[0][0], noise[0][0]]
        if rs == 0:
            for j in range(8):
                rpo[j * 8 + 0] = rpo[j * 8 + 0] + no1
                rpo[j * 8 + 1] = rpo[j * 8 + 1] + no1
        else:
            for j in range(8):
                rpo[j * 8 + 6] = rpo[j * 8 + 6] + no1
                rpo[j * 8 + 7] = rpo[j * 8 + 7] + no1
    elif rty == 2:
        no1 = [noise[0][0], noise[0][0], noise[0][0]]
        no2 = [noise[1][0], noise[1][0], noise[1][0]]
        if rs == 0:
            for j in range(8):
                rpo[0 * 8 + j] = rpo[0 * 8 + j] + no1
                rpo[1 * 8 + j] = rpo[1 * 8 + j] + no1
                rpo[j * 8 + 0] = rpo[j * 8 + 0] + no2
                rpo[j * 8 + 1] = rpo[j * 8 + 1] + no2
        else:
            for j in range(8):
                rpo[6 * 8 + j] = rpo[6 * 8 + j] + no1
                rpo[7 * 8 + j] = rpo[7 * 8 + j] + no1
                rpo[j * 8 + 6] = rpo[j * 8 + 6] + no2
                rpo[j * 8 + 7] = rpo[j * 8 + 7] + no2
    elif rty == 3:
        no1 = [noise[0][0], noise[0][0], noise[0][0]]
        no2 = [noise[1][0], noise[1][0], noise[1][0]]
        no3 = [noise[3][0], noise[3][0], noise[3][0]]
        if rs == 0:
            for j in range(8):
                rpo[0 * 8 + j] = rpo[0 * 8 + j] + no1
                rpo[1 * 8 + j] = rpo[1 * 8 + j] + no1
                rpo[j * 8 + 0] = rpo[j * 8 + 0] + no2
                rpo[j * 8 + 1] = rpo[j * 8 + 1] + no2
                rpo[j * 8 + 6] = rpo[j * 8 + 6] + no3
                rpo[j * 8 + 7] = rpo[j * 8 + 7] + no3
        else:
            for j in range(8):
                rpo[6 * 8 + j] = rpo[6 * 8 + j] + no1
                rpo[7 * 8 + j] = rpo[7 * 8 + j] + no1
                rpo[0 * 8 + j] = rpo[0 * 8 + j] + no3
                rpo[1 * 8 + j] = rpo[1 * 8 + j] + no3
                rpo[j * 8 + 6] = rpo[j * 8 + 6] + no2
                rpo[j * 8 + 7] = rpo[j * 8 + 7] + no2
    elif rty == 4:
        no1 = [noise[0][0], noise[0][0], noise[0][0]]
        no2 = [noise[1][0], noise[1][0], noise[1][0]]
        no3 = [noise[2][0], noise[2][0], noise[2][0]]
        no4 = [noise[3][0], noise[3][0], noise[3][0]]

        for j in range(8):
            rpo[0 * 8 + j] = rpo[0 * 8 + j] + no1
            rpo[1 * 8 + j] = rpo[1 * 8 + j] + no1
            rpo[j * 8 + 0] = rpo[j * 8 + 0] + no2
            rpo[j * 8 + 1] = rpo[j * 8 + 1] + no2
            rpo[j * 8 + 6] = rpo[j * 8 + 6] + no3
            rpo[j * 8 + 7] = rpo[j * 8 + 7] + no3
            rpo[6 * 8 + j] = rpo[6 * 8 + j] + no4
            rpo[7 * 8 + j] = rpo[7 * 8 + j] + no4
    else:
        rpo =rpo
    po = rpo

    target_direction = np.array([1, 1, 1])
    target_direction = normalize(target_direction)

    polygon = []
    polygon.append(po[0])
    polygon.append(po[7])
    polygon.append(po[-1])
    # 选择前两个点来计算法向量（假设多边形为三角形或四边形）
    edge1 = polygon[1] - polygon[0]
    edge2 = polygon[2] - polygon[0]
    current_normal = np.cross(edge1, edge2)

    # 计算当前法向量和目标方向之间的旋转
    current_normal = normalize(current_normal)

    # 计算旋转轴
    rotation_axis = np.cross(current_normal, target_direction)
    # if np.linalg.norm(rotation_axis) == 0:
    #     return polygon  # 当前朝向已经是目标朝向，返回原始多边形

    # 计算旋转角度
    cos_theta = np.dot(current_normal, target_direction)
    theta = np.arccos(cos_theta)  # 旋转角度

    # 旋转多边形的所有点
    rpo = np.array([rotate_point(p - [0.5, 0.5, 0.5], rotation_axis, theta) + [0.5, 0.5, 0.5] for p in po])
    po = rpo.tolist()
    x = []
    y = []
    z = []
    for poi in po:
        x.append(poi[0])
        y.append(poi[1])
        z.append(poi[2])

    mgmaxx = max(x)
    mgminx = min(x)
    mgmaxy = max(y)
    mgminy = min(y)
    mgmaxz = max(z)
    mgminz = min(z)

    po = []
    for cid in range(len(x)):
        x[cid] = ((x[cid] - mgminx) / (mgmaxx - mgminx))
        y[cid] = ((y[cid] - mgminy) / (mgmaxy - mgminy))
        z[cid] = ((z[cid] - mgminz) / (mgmaxz - mgminz))

    for poid in range(len(x)):
        po.append([x[poid], y[poid], z[poid]])


    vocelid = []

    for poi in po:
        voxel_index = get_voxel_index(poi)
        vocelid.append(voxel_index)


    return po,vocelid



