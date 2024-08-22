import argparse
import os
import sys
from pathlib import Path
import numpy as np
import math
import json

import re
# import mayavi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import copy

import cv2
import torch
numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

# cmap = plt.cm.get_cmap('hsv', 256)
# cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

def parse_header(lines):
    '''Parse header of PCD files'''
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn(f'warning: cannot understand line: {ln}')
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()

    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata

def _build_dtype(metadata):
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_ascii_pc_data(f, dtype, metadata):
    # for radar point
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    # for lidar point
    rowstep = metadata['points'] * dtype.itemsize
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    raise NotImplemented

def read_pcd(pcd_path, pts_view=False):
    # pcd = o3d.io.read_point_cloud(pcd_path)
    f = open(pcd_path, 'rb')
    header = []
    while True:
        ln = f.readline().strip()  # ln is bytes
        ln = str(ln, encoding='utf-8')
        # ln = str(ln)
        header.append(ln)
        # print(type(ln), ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or "binary_compressed"')

    points = np.concatenate([pc_data[metadata['fields'][0]][:, None],
                             pc_data[metadata['fields'][1]][:, None]], axis=-1)
    # print(points.shape)
    return points

def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data


# class TI_Process():
#
#     def __init__(self):
#         print()

# 求出两个点之间的向量角度，向量方向由点1指向点2
def getThetaOfTwoPoints(x1, y1, x2, y2):
    return math.atan2(y2-y1, x2-x1)

# 求出两个点的距离
def getDistOfTwoPoints(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))

# 在pt_set点集中找到距(p_x, p_y)最近点的id
def getClosestID(p_x, p_y, pt_set):
    id = 0
    min = 10000000
    for i in range(pt_set.shape[1]):
        dist = getDistOfTwoPoints(p_x, p_y, pt_set[0][i], pt_set[1][i])
        if dist < min:
            id = i
            min = dist
    return id

# 求出两个点集之间的平均点距
def DistOfTwoSet(set1, set2):
    loss = 0
    for i in range(set1.shape[1]):
        id = getClosestID(set1[0][i], set1[1][i], set2)
        dist = getDistOfTwoPoints(set1[0][i], set1[1][i], set2[0][id], set2[1][id])
        loss = loss + dist
    return loss/set1.shape[1]

# ICP核心代码
def ICP(sourcePoints, targetPoints):
    A = targetPoints
    B = sourcePoints

    iteration_times = 0
    dist_now = 1
    dist_improve = 1
    dist_before = DistOfTwoSet(A, B)
    while iteration_times < 10 and dist_improve > 0.001:
    # while iteration_times < 10:
        x_mean_target = A[0].mean()
        y_mean_target = A[1].mean()
        x_mean_source = B[0].mean()
        y_mean_source = B[1].mean()

        A_ = A - np.array([[x_mean_target], [y_mean_target]])
        B_ = B - np.array([[x_mean_source], [y_mean_source]])

        w_up = 0
        w_down = 0
        for i in range(A_.shape[1]):
            j = getClosestID(A_[0][i], A_[1][i], B_)
            w_up_i = A_[0][i]*B_[1][j] - A_[1][i]*B_[0][j]
            w_down_i = A_[0][i]*B_[0][j] + A_[1][i]*B_[1][j]
            w_up = w_up + w_up_i
            w_down = w_down + w_down_i

        theta = math.atan2(w_up, w_down)
        x = x_mean_target - math.cos(theta)*x_mean_source - math.sin(theta)*y_mean_source
        y = y_mean_target + math.sin(theta)*x_mean_source - math.cos(theta)*y_mean_source
        R = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])

        # B = np.matmul(R, B)
        B = np.matmul(R, B) + np.array([[x], [y]])

        iteration_times = iteration_times + 1
        dist_now = DistOfTwoSet(A, B)
        dist_improve = dist_before - dist_now
        print("迭代第"+str(iteration_times)+"次, 损失是"+str(dist_now)+",提升了"+str(dist_improve))
        dist_before = dist_now

    return B

# base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211025_2_group0013_351frames_71labeled'
# base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/tmp2'
# base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'

folders = os.listdir(base_path)
folders = sorted(folders)
frame_num = -1
plot = True
TI_points = []
pre_TI_points = []
for folder in folders:
    if 'labeled' not in folder:
        continue
    frame_num += 1
    camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
    for file in os.listdir(camera_path):
        if file[-3:] == 'png':
            img_path = os.path.join(camera_path, file)
        if file[-4:] == 'json':
            img_json = os.path.join(camera_path, file)
    print('img_path:', img_path)
    # lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
    # for file in os.listdir(lidar_path):
    #     if file[-3:] == 'pcd':
    #         pcd_lidar = os.path.join(lidar_path, file)
    #     if file[-4:] == 'json':
    #         calib_lidar = os.path.join(lidar_path, file)
    ti_path = os.path.join(base_path, folder, 'TIRadar')
    for file in os.listdir(ti_path):
        if file[-3:] == 'pcd':
            TI_pcd_path = os.path.join(ti_path, file)
        if file[-4:] == 'json':
            TI_radar_json_path = os.path.join(ti_path, file)

    TI_points = read_pcd(TI_pcd_path).T     # (N, 2)
    # print('frame_num:', frame_num)
    # print('TI_points: ', TI_points.shape)
    if frame_num != 0:
        TI_points = ICP(TI_points, pre_TI_points)

    if plot:
        draw_TI_points = TI_points.T
        plt.xlim(-40, 40)
        plt.ylim(0, 40)
        mask_pd = np.ones(draw_TI_points.shape[0], dtype=bool)
        mask_pd = np.logical_and(mask_pd, draw_TI_points[:, 1] <= 40)
        mask_pd = np.logical_and(mask_pd, draw_TI_points[:, 0] <= 40)
        mask_pd = np.logical_and(mask_pd, draw_TI_points[:, 0] >= -40)
        draw_TI_points = draw_TI_points[mask_pd, :]
        draw_TI_points = np.array(sorted(draw_TI_points, key=lambda x: x[1], reverse=False))
        # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        # cmap_name = 'my_cmap'
        # cmp = LinearSegmentedColormap.from_list(cmap_name, colors, N=40)

        cmp = ListedColormap(['b', 'g', 'r'])
        norm = BoundaryNorm(draw_TI_points[:, 1], len(draw_TI_points[:, 1]))

        for i1 in range(draw_TI_points.shape[0]):
            point = draw_TI_points[i1, :]
            # random = np.random.RandomState(int(point[1]) + 13)
            # color = random.uniform(0., 1., size=3)
            # cm = plt.cm.get_cmap('RdYlBu')
            sc = plt.scatter(point[0], point[1], s=1, c=point[1], marker='o', cmap=cmp, norm=norm)
            plt.title(folder)
            # plt.colorbar(sc)

        plt.pause(0.1)
        plt.clf()


    pre_TI_points = TI_points







