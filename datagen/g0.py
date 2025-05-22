import math
import numpy as np
from geomdl import BSpline
import os
import random
import torch

import matplotlib

matplotlib.use('TkAgg')


def dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def calculate_gaussian_curvature(u, v, srf):
    # 计算偏导数
    skl1 = srf.derivatives(u, v, 2)

    norm = normalize(cross_product(skl1[1][0], skl1[0][1]))

    dx_du = skl1[1][0][0]
    dy_du = skl1[1][0][1]
    dz_du = skl1[1][0][2]

    dx_dv = skl1[0][1][0]
    dy_dv = skl1[0][1][1]
    dz_dv = skl1[0][1][2]

    E = dot_product(skl1[1][0], skl1[1][0])
    F = dot_product(skl1[1][0], skl1[0][1])
    G = dot_product(skl1[0][1], skl1[0][1])

    # 计算法向量

    # 计算第二基本形式
    L = dot_product(skl1[2][0], norm)
    M = dot_product(skl1[1][1], norm)
    N = dot_product(skl1[0][2], norm)

    # 计算高斯曲率 K
    K = (L * N - M ** 2) / (E * G - F ** 2)

    return K


def get_voxel_index(point):
    """
    获取给定点所对应的体素索引。

    参数:
    - point: 一个包含三个元素的列表或元组，表示点的坐标 (x, y, z)

    返回:
    - voxel_index: 体素的唯一索引
    """
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


def cross_product(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]



def normalize(vector):
    magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    normalized_vector = [component / magnitude for component in vector]

    return normalized_vector


aa = ['bdata1.txt', 'bdata10.txt', 'bdata102.txt', 'bdata104.txt', 'bdata105.txt', 'bdata109.txt', 'bdata11.txt', 'bdata110.txt', 'bdata111.txt', 'bdata112.txt', 'bdata119.txt', 'bdata12.txt', 'bdata120.txt', 'bdata121.txt', 'bdata124.txt', 'bdata125.txt', 'bdata129.txt', 'bdata13.txt', 'bdata130.txt', 'bdata134.txt', 'bdata139.txt', 'bdata14.txt', 'bdata140.txt', 'bdata141.txt', 'bdata142.txt', 'bdata143.txt', 'bdata144.txt', 'bdata148.txt', 'bdata150.txt', 'bdata155.txt', 'bdata157.txt', 'bdata158.txt', 'bdata159.txt', 'bdata166.txt', 'bdata167.txt', 'bdata169.txt', 'bdata170.txt', 'bdata172.txt', 'bdata175.txt', 'bdata178.txt', 'bdata18.txt', 'bdata182.txt', 'bdata187.txt', 'bdata189.txt', 'bdata19.txt', 'bdata191.txt', 'bdata193.txt', 'bdata194.txt', 'bdata195.txt', 'bdata197.txt', 'bdata2.txt', 'bdata207.txt', 'bdata21.txt', 'bdata211.txt', 'bdata212.txt', 'bdata217.txt', 'bdata218.txt', 'bdata219.txt', 'bdata22.txt', 'bdata220.txt', 'bdata221.txt', 'bdata223.txt', 'bdata224.txt', 'bdata225.txt', 'bdata226.txt', 'bdata227.txt', 'bdata230.txt', 'bdata231.txt', 'bdata232.txt', 'bdata234.txt', 'bdata24.txt', 'bdata240.txt', 'bdata242.txt', 'bdata243.txt', 'bdata244.txt', 'bdata245.txt', 'bdata248.txt', 'bdata251.txt', 'bdata255.txt', 'bdata256.txt', 'bdata259.txt', 'bdata260.txt', 'bdata264.txt', 'bdata267.txt', 'bdata27.txt', 'bdata271.txt', 'bdata272.txt', 'bdata273.txt', 'bdata277.txt', 'bdata28.txt', 'bdata280.txt', 'bdata284.txt', 'bdata285.txt', 'bdata286.txt', 'bdata287.txt', 'bdata288.txt', 'bdata289.txt', 'bdata290.txt', 'bdata291.txt']
bb = ['bdata293.txt', 'bdata295.txt', 'bdata296.txt', 'bdata298.txt', 'bdata300.txt', 'bdata301.txt', 'bdata303.txt', 'bdata304.txt', 'bdata305.txt', 'bdata306.txt', 'bdata307.txt', 'bdata309.txt', 'bdata31.txt', 'bdata312.txt', 'bdata313.txt', 'bdata319.txt', 'bdata32.txt', 'bdata321.txt', 'bdata322.txt', 'bdata323.txt', 'bdata324.txt', 'bdata325.txt', 'bdata331.txt', 'bdata332.txt', 'bdata333.txt', 'bdata334.txt', 'bdata336.txt', 'bdata337.txt']
cc = ['bdata337.txt', 'bdata339.txt', 'bdata34.txt', 'bdata341.txt', 'bdata342.txt', 'bdata343.txt', 'bdata344.txt', 'bdata345.txt', 'bdata346.txt', 'bdata347.txt', 'bdata349.txt', 'bdata351.txt', 'bdata352.txt', 'bdata356.txt', 'bdata359.txt', 'bdata36.txt', 'bdata360.txt', 'bdata361.txt','subbdata1081.txt', 'subbdata1085.txt', 'subbdata1087.txt', 'subbdata1088.txt', 'subbdata1092.txt', 'subbdata1098.txt', 'subbdata110.txt']
dd = ['bdata360.txt', 'bdata361.txt', 'bdata362.txt', 'bdata366.txt', 'bdata371.txt', 'bdata372.txt', 'bdata373.txt', 'bdata379.txt', 'bdata380.txt', 'bdata382.txt', 'bdata386.txt', 'bdata387.txt', 'bdata388.txt', 'bdata389.txt', 'bdata39.txt', 'bdata40.txt', 'bdata46.txt', 'bdata47.txt', 'bdata51.txt', 'bdata54.txt', 'bdata55.txt', 'bdata58.txt', 'bdata59.txt', 'bdata6.txt', 'bdata65.txt', 'bdata67.txt', 'bdata69.txt', 'bdata72.txt', 'bdata8.txt', 'bdata88.txt', 'bdata89.txt', 'bdata9.txt', 'bdata94.txt', 'cdata0.txt', 'cdata1.txt', 'cdata10.txt', 'cdata11.txt', 'cdata12.txt', 'cdata13.txt', 'cdata14.txt', 'cdata15.txt', 'cdata16.txt', 'cdata18.txt', 'cdata2.txt']
ee = ['cdata18.txt', 'cdata2.txt', 'cdata22.txt', 'cdata23.txt', 'cdata24.txt', 'cdata32.txt', 'cdata33.txt', 'cdata37.txt', 'cdata4.txt', 'cdata40.txt', 'cdata41.txt', 'cdata48.txt', 'cdata5.txt', 'cdata51.txt', 'cdata52.txt', 'cdata53.txt', 'cdata6.txt', 'cdata7.txt', 'subbdata1010.txt', 'subbdata102.txt', 'subbdata1020.txt', 'subbdata1023.txt', 'subbdata1024.txt', 'subbdata103.txt', 'subbdata1031.txt', 'subbdata1032.txt', 'subbdata1033.txt', 'subbdata1034.txt', 'subbdata1037.txt', 'subbdata1038.txt','subbdata1050.txt', 'subbdata1051.txt', 'subbdata1052.txt', 'subbdata106.txt', 'subbdata1060.txt', 'subbdata1061.txt', 'subbdata1062.txt', 'subbdata1072.txt']
ff = ['subbdata1465.txt','subbdata139.txt', 'subbdata140.txt', 'subbdata142.txt', 'subbdata1421.txt', 'subbdata1426.txt', 'subbdata1438.txt', 'subbdata1440.txt', 'subbdata1442.txt','subbdata1361.txt', 'subbdata137.txt', 'subbdata138.txt','subbdata1325.txt', 'subbdata1335.txt', 'subbdata1343.txt', 'subbdata1344.txt', 'subbdata1350.txt','subbdata1232.txt', 'subbdata1236.txt','subbdata1232.txt','subbdata1104.txt', 'subbdata1114.txt', 'subbdata112.txt', 'subbdata1122.txt', 'subbdata1125.txt','subbdata1128.txt', 'subbdata114.txt', 'subbdata1148.txt', 'subbdata116.txt', 'subbdata1162.txt','subbdata117.txt', 'subbdata118.txt', 'subbdata1189.txt', 'subbdata12.txt', 'subbdata120.txt', 'subbdata1200.txt', 'subbdata1212.txt', 'subbdata1219.txt', 'subbdata122.txt', 'subbdata1220.txt', 'subbdata1226.txt']
gg = ['subcdata328.txt', 'subcdata366.txt', 'subcdata41.txt', 'subcdata43.txt', 'subcdata430.txt', 'subcdata470.txt', 'subcdata471.txt', 'subcdata49.txt', 'subcdata494.txt', 'subcdata50.txt', 'subcdata519.txt', 'subcdata522.txt', 'subcdata523.txt', 'subcdata53.txt', 'subcdata537.txt', 'subcdata543.txt', 'subcdata615.txt', 'subcdata635.txt', 'subcdata648.txt', 'subcdata650.txt', 'subcdata687.txt', 'subcdata690.txt', 'subcdata691.txt', 'subcdata697.txt', 'subcdata704.txt', 'subcdata720.txt', 'subcdata727.txt', 'subcdata748.txt', 'subcdata754.txt', 'subcdata760.txt', 'subcdata765.txt', 'subcdata795.txt', 'subcdata8.txt', 'subcdata802.txt', 'subcdata881.txt', 'subcdata9.txt', 'subcdata92.txt', 'subhdata10.txt','subcdata21.txt', 'subcdata213.txt', 'subcdata218.txt', 'subcdata224.txt', 'subcdata229.txt', 'subcdata243.txt', 'subcdata248.txt', 'subcdata25.txt', 'subcdata250.txt', 'subcdata262.txt', 'subcdata27.txt', 'subcdata280.txt', 'subcdata281.txt', 'subcdata282.txt', 'subcdata294.txt', 'subcdata297.txt', 'subcdata305.txt', 'subcdata307.txt','subcdata1208.txt', 'subcdata1210.txt', 'subcdata1211.txt', 'subcdata1212.txt', 'subcdata1225.txt', 'subcdata1229.txt', 'subcdata1235.txt', 'subcdata1238.txt', 'subcdata1239.txt', 'subcdata124.txt', 'subcdata1243.txt', 'subcdata1259.txt', 'subcdata126.txt', 'subcdata1260.txt', 'subcdata1261.txt', 'subcdata1263.txt', 'subcdata1264.txt', 'subcdata1266.txt', 'subcdata1267.txt', 'subcdata1268.txt', 'subcdata1270.txt', 'subcdata1283.txt', 'subcdata1284.txt', 'subcdata1285.txt', 'subcdata1286.txt', 'subcdata1289.txt', 'subcdata1290.txt', 'subcdata1294.txt', 'subcdata1306.txt', 'subcdata1313.txt', 'subcdata1321.txt', 'subcdata1322.txt', 'subcdata133.txt', 'subcdata134.txt', 'subcdata1341.txt', 'subcdata1342.txt', 'subcdata1343.txt', 'subcdata1344.txt', 'subcdata1346.txt', 'subcdata1349.txt', 'subcdata135.txt', 'subcdata1352.txt', 'subcdata1356.txt', 'subcdata1361.txt', 'subcdata138.txt', 'subcdata14.txt', 'subcdata1471.txt', 'subcdata1475.txt', 'subcdata1483.txt', 'subcdata1486.txt', 'subcdata1488.txt', 'subcdata1489.txt', 'subcdata1498.txt', 'subcdata1507.txt', 'subcdata1508.txt', 'subcdata1510.txt', 'subcdata1519.txt', 'subcdata1521.txt', 'subcdata1531.txt', 'subcdata1535.txt', 'subcdata1537.txt', 'subcdata1540.txt', 'subcdata166.txt', 'subcdata168.txt', 'subcdata169.txt', 'subcdata174.txt', 'subcdata175.txt', 'subcdata176.txt', 'subcdata18.txt', 'subcdata20.txt', 'subcdata201.txt', 'subcdata205.txt', 'subcdata206.txt','subcdata12.txt', 'subcdata120.txt', 'subcdata1207.txt', 'subcdata1208.txt','subcdata1196.txt', 'subcdata12.txt','subbdata910.txt', 'subbdata922.txt', 'subbdata928.txt', 'subbdata935.txt', 'subbdata974.txt', 'subbdata983.txt', 'subbdata984.txt', 'subcdata1.txt', 'subcdata1030.txt', 'subcdata1063.txt', 'subcdata1079.txt', 'subcdata1083.txt', 'subcdata1084.txt', 'subcdata1092.txt', 'subcdata1130.txt', 'subcdata1131.txt',
      'subbdata798.txt', 'subbdata799.txt', 'subbdata80.txt', 'subbdata858.txt', 'subbdata878.txt', 'subbdata883.txt', 'subbdata892.txt', 'subbdata907.txt', 'subbdata91.txt','subbdata767.txt', 'subbdata776.txt','subbdata691.txt', 'subbdata730.txt', 'subbdata763.txt','subbdata1465.txt', 'subbdata1495.txt', 'subbdata1521.txt', 'subbdata1528.txt', 'subbdata1537.txt', 'subbdata1543.txt', 'subbdata1545.txt', 'subbdata1549.txt', 'subbdata1554.txt', 'subbdata1556.txt', 'subbdata156.txt', 'subbdata1568.txt', 'subbdata1574.txt', 'subbdata1576.txt', 'subbdata1578.txt', 'subbdata1579.txt', 'subbdata1580.txt', 'subbdata1582.txt', 'subbdata1585.txt', 'subbdata1586.txt', 'subbdata1588.txt', 'subbdata1594.txt', 'subbdata1602.txt', 'subbdata1603.txt', 'subbdata1605.txt', 'subbdata1608.txt', 'subbdata1609.txt', 'subbdata1614.txt', 'subbdata1616.txt', 'subbdata1617.txt', 'subbdata1618.txt', 'subbdata1619.txt', 'subbdata162.txt', 'subbdata170.txt', 'subbdata181.txt', 'subbdata190.txt', 'subbdata194.txt', 'subbdata199.txt', 'subbdata211.txt', 'subbdata219.txt', 'subbdata232.txt', 'subbdata235.txt', 'subbdata249.txt', 'subbdata253.txt', 'subbdata270.txt', 'subbdata272.txt', 'subbdata308.txt', 'subbdata309.txt', 'subbdata33.txt', 'subbdata340.txt', 'subbdata346.txt', 'subbdata354.txt', 'subbdata356.txt', 'subbdata382.txt', 'subbdata391.txt', 'subbdata406.txt', 'subbdata422.txt', 'subbdata429.txt', 'subbdata43.txt', 'subbdata440.txt', 'subbdata450.txt', 'subbdata467.txt', 'subbdata491.txt', 'subbdata492.txt', 'subbdata521.txt', 'subbdata524.txt', 'subbdata526.txt', 'subbdata53.txt', 'subbdata544.txt', 'subbdata570.txt', 'subbdata573.txt', 'subbdata574.txt', 'subbdata575.txt', 'subbdata646.txt', 'subbdata647.txt']


sur = aa+bb+cc+dd+ee+ff+gg


filepaths = os.listdir('../curve')

pcu = []
for filepa in filepaths:

    f = open('../curve/' + filepa, encoding='utf-8')
    uv = []
    for line in f:
        stuv = line.split("/")
        uv.append([float(stuv[0]), float(stuv[1])])
    if (len(uv) != 297):
        print(filepa)
        continue
    pcu.append(uv)

filenames = os.listdir("../manyface")

srf = BSpline.Surface()
srf.degree_u = 3
srf.degree_v = 3
srf.ctrlpts_size_u = 8
srf.ctrlpts_size_v = 8

orderlist = np.random.choice(20 * len(filenames), 20 * len(filenames), replace=False)
fii = 0


for fiId in range(len(filenames)):
    curnum = 1
    if filenames in sur:
        curnum = 10
    else:
        rannum = random.randint(1, 10)
        if rannum==5 or rannum==6:
            continue
    f = open('../manyface/' + filenames[fiId], encoding='utf-8')
    po = []
    x = []
    y = []
    z = []
    for line in f:
        sa = line.split("/")
        for i in range(8):
            x.append(float(sa[i * 3]))
            y.append(float(sa[i * 3 + 1]))
            z.append(float(sa[i * 3 + 2]))
            # po.append([float(sa[i * 3]), float(sa[i * 3 + 1]), float(sa[i * 3 + 2])])
    mgmaxx = max(x)
    mgminx = min(x)
    mgmaxy = max(y)
    mgminy = min(y)
    mgmaxz = max(z)
    mgminz = min(z)

    mag = max(mgmaxx, mgmaxy, mgmaxz)
    mig = min(mgminx, mgminy, mgminz)

    midx = 0.5 - (mgmaxx + mgminx) / 2
    midy = 0.5 - (mgmaxy + mgminy) / 2
    midz = 0.5 - (mgmaxz + mgminz) / 2

    for cid in range(len(x)):
        x[cid] = (x[cid] + midx)
        y[cid] = (y[cid] + midy)
        z[cid] = (z[cid] + midz)

    mgmaxx = max(x)
    mgminx = min(x)
    mgmaxy = max(y)
    mgminy = min(y)
    mgmaxz = max(z)
    mgminz = min(z)

    mag = max(mgmaxx, mgmaxy, mgmaxz)
    mig = min(mgminx, mgminy, mgminz)

    for cid in range(len(x)):
        x[cid] = ((x[cid] - mig) / (mag - mig))
        y[cid] = ((y[cid] - mig) / (mag - mig))
        z[cid] = ((z[cid] - mig) / (mag - mig))
    for poid in range(len(x)):
        po.append([x[poid], y[poid], z[poid]])

    vocelid = []

    for poi in po:
        voxel_index = get_voxel_index(poi)
        vocelid.append(voxel_index)

    #
    uvi = np.random.choice(400, curnum, replace=False)
    # uvi = random.sample(range(141,160),2)
    print(fiId)
    # uvi = [0]
    for uvid in range(curnum):

        knotu = []
        knotv = []

        for i in range(4):
            knotu.append(0)
            knotv.append(0)

        ranknotu = []
        ranknotv = []
        for i in range(4):
            ranknotu.append(random.uniform(0, 1))
            ranknotv.append(random.uniform(0, 1))
        ranknotu.sort()
        ranknotv.sort()
        for i in range(4):
            knotu.append(ranknotu[i])
            knotv.append(ranknotv[i])

        for i in range(4):
            knotu.append(1)
            knotv.append(1)

        srf.knotvector_u = knotu
        srf.knotvector_v = knotv

        srf.ctrlpts = po

        # if uvid ==0:
        #     u = np.linspace(0.1, 0.9, 50)  # 参数范围
        #     v = np.linspace(0.1, 0.9, 50)
        #     allk = []
        #     for ui in range(50):
        #         K = calculate_gaussian_curvature(u[ui], v[ui], srf)
        #         allk.append(abs(K))
        #     if max(allk)<0.5:
        #         print("高斯曲率:", max(allk))
        #         break

        distanceu = []
        distancev = []
        for i in range(4, 8):
            distanceu.append(knotu[i])
            distancev.append(knotv[i])

        geolistx = []
        geolisty = []
        geolistz = []
        geolist2 = []
        uuvid = 0 + uvid
        for poi in range(len(pcu[uvi[uuvid]])):

            if pcu[uvi[uuvid]][poi][0] < 0:
                pcu[uvi[uuvid]][poi][0] = 0
            if pcu[uvi[uuvid]][poi][1] < 0:
                pcu[uvi[uuvid]][poi][1] = 0

            if pcu[uvi[uuvid]][poi][0] > 1:
                pcu[uvi[uuvid]][poi][0] = 1
            if pcu[uvi[uuvid]][poi][1] > 1:
                pcu[uvi[uuvid]][poi][1] = 1
            u = pcu[uvi[uuvid]][poi][0]
            v = pcu[uvi[uuvid]][poi][1]
            # skl1 = srf.derivatives(pcu[uvi[uuvid]][poi][0], pcu[uvi[uuvid]][poi][1], 2)
            skl1 = srf.evaluate_single((pcu[uvi[uuvid]][poi][0], pcu[uvi[uuvid]][poi][1]))

            # ut = 0
            # vt = 0
            # if (poi + 1) % 99 == 0:
            #     ut = u - pcu[uvi[uuvid]][poi - 1][0]
            #     vt = v - pcu[uvi[uuvid]][poi - 1][1]
            # else:
            #     ut = -u + pcu[uvi[uuvid]][poi + 1][0]
            #     vt = -v + pcu[uvi[uuvid]][poi + 1][1]
            #
            # curvex = vt * vt * skl1[2][0][0] - 2 * ut * vt * skl1[1][1][0] + ut * ut * skl1[0][2][0]
            # curvey = vt * vt * skl1[2][0][1] - 2 * ut * vt * skl1[1][1][1] + ut * ut * skl1[0][2][1]
            # curvez = vt * vt * skl1[2][0][2] - 2 * ut * vt * skl1[1][1][2] + ut * ut * skl1[0][2][2]

            # norm = normalize(cross_product(skl1[1][0], skl1[0][1]))
            pono = []
            pono.append(skl1[0])
            pono.append(skl1[1])
            pono.append(skl1[2])
            # pono.append(norm[0])
            # pono.append(norm[1])
            # pono.append(norm[2])
            # pono.append(curvex)
            # pono.append(curvey)
            # pono.append(curvez)
            geolist2.append(pono)
            # geolist2.append(norm)

        #
        # for cid in range(len(geolistx)):
        #     geolistx[cid] = (geolistx[cid]- minx)/( maxx -  minx)
        #     geolisty[cid] = (geolisty[cid] -  miny) / ( maxy -  miny)
        #     geolistz[cid] = (geolistz[cid] -  minz) / ( maxz -  minz)

        geolist2 = torch.tensor(geolist2)
        # poin = geolist2[:,::3]
        # T = poin[1:] - poin[:-1]
        # T_norm = torch.norm(T, dim=1, keepdim=True).reshape(-1)
        #
        # count_of_zeros = (T_norm == 0).sum().item()
        # if count_of_zeros>6:
        #     break

        uvlist = pcu[uvi[uvid]]

        geolist = []

        uvtensor = torch.tensor(uvlist).reshape(-1)

        torch.save(geolist2, '../g0data/input/' + str(fii) + '.pt')
        pointtensor = torch.tensor(po).reshape(-1)
        voxet = torch.tensor(vocelid)
        knotutensor = torch.tensor(distanceu).reshape(-1)
        knotvtensor = torch.tensor(distancev).reshape(-1)
        pointknot = torch.cat((knotutensor, knotvtensor, uvtensor, pointtensor))
        torch.save(pointknot, '../g0data/output/' + str(fii) + '.pt')
        torch.save(voxet, '../g0data/voutput/' + str(fii) + '.pt')

        fii += 1