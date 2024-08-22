
import os


import sys
import argparse
import numpy as np
import cv2
import torch
import openpyxl
from dataset.self_dataset2 import self_dataset2
import matplotlib.pyplot as plt

from TiProcess.BEV_TI import OCUpts2rbev

def pointcloud_transform(pointcloud, transform_matrix):
    '''
        transform pointcloud from coordinate1 to coordinate2 according to transform_matrix
    :param pointcloud: (x, y, z, id, classes, ...) 8*N
    :param transform_matrix: 4*4
    :return pointcloud_transformed: (x, y, z, ...)
    '''
    n_points = pointcloud.shape[1]
    xyz = pointcloud[0:3, :]
    xyz1 = np.vstack((xyz, np.ones((1, n_points)))) # 4*N
    xyz1_transformed = np.matmul(transform_matrix, xyz1)    # 4*N
    pointcloud_transformed = np.vstack((
        xyz1_transformed[:3, :],
        pointcloud[3:, :]
    ))  # 5*N
    return pointcloud_transformed

def lidar2img(lidar_json, OCU_json, cam_json, curr_label):
    # classes 2 means car
    # Get LiDAR annotation points
    lidar_raw_anno = curr_label
    x, y, z, gt_id, gt_classes = [], [], [], [], []
    for idx in range(len(lidar_raw_anno)):
        if lidar_raw_anno[idx]['class'] != 'car':
            continue
        # if lidar_raw_anno[idx]['x'] >= 40 or lidar_raw_anno[idx]['x'] <= 0 \
        #         or lidar_raw_anno[idx]['y'] >= 20 or lidar_raw_anno[idx]['y'] <= -5:
        #     continue
        x.append(lidar_raw_anno[idx]['x'])
        y.append(lidar_raw_anno[idx]['y'])
        z.append(lidar_raw_anno[idx]['z'])
        gt_id.append(int(lidar_raw_anno[idx]['object_id']))
        gt_classes.append(2)

    if len(x) > 0:
        # Lidar_anno_points = np.vstack((x, y, z)).T   # (N, 3)
        Lidar_anno_points = np.vstack((x, y, z, gt_id, gt_classes))  # (5, N)
    else:
        Lidar_anno_points = np.array([])
        return Lidar_anno_points

    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera0_TransformMatrix = np.array(lidar_json['VelodyneLidar_to_LeopardCamera0_extrinsic']) # 4*4
    OCU_to_LeopardCamera1_TransformMatrix = np.array(OCU_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])   # 4*4
    VelodyneLidar2OCU_TransformMatrix = np.matmul(
        np.linalg.inv(OCU_to_LeopardCamera1_TransformMatrix),
        VelodyneLidar_to_LeopardCamera0_TransformMatrix
    )   # 4*4

    LeopardCamera0_IntrinsicMatrix = np.array(cam_json['intrinsic'])  # 3*3
    LeopardCamera0_IntrinsicMatrix_34 = np.hstack(
        (LeopardCamera0_IntrinsicMatrix, np.array([[0, 0, 0]]).reshape(3, 1)))  # 3*4
    VelodyneLidar_to_LeopardCamera0 = np.matmul(LeopardCamera0_IntrinsicMatrix_34,
                                                VelodyneLidar_to_LeopardCamera0_TransformMatrix)  # 3*4

    Lidar2OCU_anno_points = pointcloud_transform(Lidar_anno_points, VelodyneLidar2OCU_TransformMatrix)  # (5, N)

    mask0 = np.ones(Lidar2OCU_anno_points.shape[1], dtype=bool)
    mask0 = np.logical_and(mask0, Lidar2OCU_anno_points[0, :] >= -10)
    mask0 = np.logical_and(mask0, Lidar2OCU_anno_points[0, :] <= 20)
    mask0 = np.logical_and(mask0, Lidar2OCU_anno_points[2, :] >= 0)
    mask0 = np.logical_and(mask0, Lidar2OCU_anno_points[2, :] <= 40)
    Lidar2OCU_anno_points2 = Lidar2OCU_anno_points[:, mask0]
    Lidar_anno_points_filter = Lidar_anno_points[:, mask0]
    # final_x, final_y, final_z, final_id, final_classes = [], [], [], [], []
    # for i in range(Lidar2OCU_anno_points.shape[1]):
    #     tmp_x = Lidar2OCU_anno_points[0][i]
    #     tmp_y = Lidar2OCU_anno_points[1][i]
    #     tmp_z = Lidar2OCU_anno_points[2][i]
    #
    #     if tmp_x >= 20 or tmp_x <= -10 or tmp_z >= 40 or tmp_z <= 0:
    #         continue
    #     final_x.append(tmp_x)
    #     final_y.append(tmp_y)
    #     final_z.append(tmp_z)
    #     final_id.append(Lidar2OCU_anno_points[3][i])
    #     final_classes.append(Lidar2OCU_anno_points[4][i])
    #
    # if len(final_x) > 0:
    #     Lidar_final_anno_points = np.vstack((final_id, final_x, final_y, final_z, final_classes))  # (5, N)
    # else:
    #     Lidar_final_anno_points = np.array([])
    #     return Lidar_final_anno_points


    Lidar_anno_points_xyz = Lidar_anno_points[0:3, :]  # 3*N
    ones = np.ones((1, Lidar_anno_points_xyz.shape[1]))  # 1*N
    Lidar_anno_points_xyz1 = np.vstack((Lidar_anno_points_xyz, ones))  # 4*N
    img_uv1 = np.matmul(VelodyneLidar_to_LeopardCamera0, Lidar_anno_points_xyz1) # 3*N
    img_uv1[0, :] = img_uv1[0, :] / img_uv1[2, :]
    img_uv1[1, :] = img_uv1[1, :] / img_uv1[2, :]
    # mask0 = np.ones(img_uv1.shape[1], dtype=bool)
    # mask0 = np.logical_and(mask0, img_uv1[:, 0] >= 0)
    # mask0 = np.logical_and(mask0, img_uv1[:, 0] <= 1111)
    # mask0 = np.logical_and(mask0, img_uv1[:, 1] >= 0)
    # mask0 = np.logical_and(mask0, img_uv1[:, 1] <= 1111)
    # img_uv = img_uv1[mask0, :]

    return img_uv1, Lidar2OCU_anno_points2, Lidar2OCU_anno_points

# Lidar_points (N, 5)
def lidar2img_filter(lidar_json, OCU_json, cam_json, img_width, img_height, extendGT):
    # classes 2 means car
    # Get LiDAR annotation points

    lidar_raw_anno = extendGT
    x, y, z, l, w, h, gt_id, gt_classes = [], [], [], [], [], [], [], []
    for idx in range(len(lidar_raw_anno)):
        if lidar_raw_anno[idx]['class'] != 'car':
            continue
        x.append(lidar_raw_anno[idx]['x'])
        y.append(lidar_raw_anno[idx]['y'])
        z.append(lidar_raw_anno[idx]['z'])
        l.append(lidar_raw_anno[idx]['l'])
        w.append(lidar_raw_anno[idx]['w'])
        h.append(lidar_raw_anno[idx]['h'])
        gt_id.append(int(lidar_raw_anno[idx]['object_id']))
        gt_classes.append(2)
    if len(x) > 0:
        # Lidar_anno_points = np.vstack((x, y, z)).T   # (N, 3)
        Lidar_anno_points = np.vstack((x, y, z, l, w, h, gt_id, gt_classes))  # (8, N)
    else:
        return np.array([])

    # # Lidar Points count filter
    # anno_num = Lidar_anno_points.shape[1]
    # boundings = []
    # # boundings = np.zeros(shape=(0, 6))
    # for i in range(anno_num):
    #     tmp = [Lidar_anno_points[0][i] - Lidar_anno_points[3][i], Lidar_anno_points[0][i] + Lidar_anno_points[3][i],
    #            Lidar_anno_points[1][i] - Lidar_anno_points[4][i], Lidar_anno_points[1][i] + Lidar_anno_points[4][i],
    #            Lidar_anno_points[2][i] - Lidar_anno_points[5][i], Lidar_anno_points[2][i] + Lidar_anno_points[5][i]]
    #     boundings.append(tmp)
    #     # tmp = np.array(tmp).reshape(1, 6)
    #     # boundings = np.vstack((boundings, tmp))
    # boundings = np.array(boundings) # (N, 6)
    # n_lidar_points = len(Lidar_points)
    #
    # inbbox_points = np.zeros(anno_num)
    # for i in range(anno_num):
    #     mask = np.ones(n_lidar_points, dtype=bool)
    #     mask = np.logical_and(mask, Lidar_points[:, 0] > boundings[i, 0])
    #     mask = np.logical_and(mask, Lidar_points[:, 0] < boundings[i, 1])
    #     mask = np.logical_and(mask, Lidar_points[:, 1] > boundings[i, 2])
    #     mask = np.logical_and(mask, Lidar_points[:, 1] < boundings[i, 3])
    #     mask = np.logical_and(mask, Lidar_points[:, 2] > boundings[i, 4])
    #     mask = np.logical_and(mask, Lidar_points[:, 2] < boundings[i, 5])
    #     inbbox_points[i] = sum(mask)
    #
    # # print('inbbox_points:', inbbox_points, '\n')
    # mask = np.ones(anno_num, dtype=bool)
    # mask = np.logical_and(mask, inbbox_points[:] >= 50)
    # Lidar_anno_points = Lidar_anno_points[:, mask]

    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera0_TransformMatrix = np.array(lidar_json['VelodyneLidar_to_LeopardCamera0_extrinsic']) # 4*4
    OCU_to_LeopardCamera0_TransformMatrix = np.array(OCU_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])   # 4*4
    VelodyneLidar2OCU_TransformMatrix = np.matmul(
        np.linalg.inv(OCU_to_LeopardCamera0_TransformMatrix),
        VelodyneLidar_to_LeopardCamera0_TransformMatrix
    )   # 4*4

    # LiDAR to image
    LeopardCamera0_IntrinsicMatrix = np.array(cam_json['intrinsic'])  # 3*3
    LeopardCamera0_IntrinsicMatrix_34 = np.hstack(
        (LeopardCamera0_IntrinsicMatrix, np.array([[0, 0, 0]]).reshape(3, 1)))  # 3*4
    VelodyneLidar_to_LeopardCamera0 = np.matmul(LeopardCamera0_IntrinsicMatrix_34,
                                                VelodyneLidar_to_LeopardCamera0_TransformMatrix)  # 3*4

    # FOV Filter
    Lidar_anno_points_xyz = Lidar_anno_points[0:3, :]  # 3*N
    ones = np.ones((1, Lidar_anno_points_xyz.shape[1]))  # 1*N
    Lidar_anno_points_xyz1 = np.vstack((Lidar_anno_points_xyz, ones))  # 4*N
    img_uv1 = np.matmul(VelodyneLidar_to_LeopardCamera0, Lidar_anno_points_xyz1)  # 3*N
    img_uv1[0, :] = img_uv1[0, :] / img_uv1[2, :]
    img_uv1[1, :] = img_uv1[1, :] / img_uv1[2, :]
    # if Lidar_uv1[0, i] < 0 or Lidar_uv1[0, i] > img_width or Lidar_uv1[1, i] < 0 or Lidar_uv1[1, i] > img_height:
    #     num += 1
    mask0 = np.ones(img_uv1.shape[1], dtype=bool)
    # mask0 = np.logical_and(mask0, img_uv1[0, :] >= 225-10)
    # mask0 = np.logical_and(mask0, img_uv1[0, :] <= 735+10)
    mask0 = np.logical_and(mask0, img_uv1[0, :] >= 0)
    mask0 = np.logical_and(mask0, img_uv1[0, :] < img_width)
    mask0 = np.logical_and(mask0, img_uv1[1, :] >= 0)
    mask0 = np.logical_and(mask0, img_uv1[1, :] < img_height)
    # img_uv = img_uv1[:, mask0]

    # distance Filter
    Lidar2OCU_anno_points = pointcloud_transform(Lidar_anno_points, VelodyneLidar2OCU_TransformMatrix)  # (8, N)
    mask1 = np.ones(Lidar2OCU_anno_points.shape[1], dtype=bool)
    mask1 = np.logical_and(mask1, Lidar2OCU_anno_points[0, :] >= -10)
    mask1 = np.logical_and(mask1, Lidar2OCU_anno_points[0, :] <= 10)
    mask1 = np.logical_and(mask1, Lidar2OCU_anno_points[2, :] >= 0)
    mask1 = np.logical_and(mask1, Lidar2OCU_anno_points[2, :] <= 50)
    mask = np.logical_and(mask0, mask1)
    Lidar2OCU_anno_points2 = Lidar2OCU_anno_points[:, mask]# (5, N)
    # Lidar_anno_points_filter = Lidar_anno_points[:, mask0]



    final_x, final_y, final_z, final_id, final_classes = [], [], [], [], []
    for i in range(Lidar2OCU_anno_points2.shape[1]):
        tmp_x = Lidar2OCU_anno_points2[0][i]
        tmp_y = Lidar2OCU_anno_points2[1][i]
        tmp_z = Lidar2OCU_anno_points2[2][i]

        # if tmp_x >= 20 or tmp_x <= -10 or tmp_z >= 40 or tmp_z <= 0:
        #     continue
        final_x.append(tmp_x)
        final_y.append(tmp_z)
        final_z.append(tmp_y)
        final_id.append(Lidar2OCU_anno_points2[6][i])
        final_classes.append(Lidar2OCU_anno_points2[7][i])

    if len(final_x) > 0:
        Lidar_final_anno_points = np.vstack((final_id, final_x, final_y, final_z, final_classes)).T  # (N, 5)
    else:
        return np.array([])
    # return img_uv1, Lidar2OCU_anno_points2, Lidar2OCU_anno_points
    return Lidar_final_anno_points

def run():
    file_path = '/media/personal_data/zhangq/RadarRGBFusionNet2_20231128/GroupPath/20230516.xlsx'
    DataPath = openpyxl.load_workbook(file_path)
    ws = DataPath.active
    groups_excel = ws['A']
    datasets_path = []
    for cell in groups_excel:
        datasets_path.append(cell.value)
    for i in range(len(datasets_path)):
        source = os.path.join('/media/personal_data/zhangq/RadarRGBFusionNet2_20231128/dataset/datas', datasets_path[i])
        labels_path = os.path.join('/media/personal_data/zhangq/RadarRGBFusionNet2_20231128/dataset/labels', datasets_path[i])

    imgsz = [640]
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    dataset = self_dataset2(source, labels_path, img_size=imgsz)

    for frame_idx, (rgb_img0, img, rgb_json, OCU_data, OCU_json, lidar_json, curr_label) in enumerate(dataset):
        print('frame_num:', frame_idx)
        img_width = rgb_img0.shape[1]
        img_height = rgb_img0.shape[0]
        if frame_idx == 138:
            print()
        Lidar_uv1, Lidar_anno_xyz_filter, Lidar_anno_xyz = lidar2img(lidar_json, OCU_json, rgb_json, curr_label)  # 3*N

        num = 0
        for i in range(Lidar_uv1.shape[1]):
            if Lidar_uv1[0, i] < 0 or Lidar_uv1[0, i] > img_width or Lidar_uv1[1, i] < 0 or Lidar_uv1[
                1, i] > img_height:
                num += 1
        if num > 0:
            # print('didididididididididididididididi')
            print('num', num, '\n')

        for i in range(Lidar_uv1.shape[1]):
            cv2.circle(rgb_img0, (int(Lidar_uv1[0, i]), int(Lidar_uv1[1, i])), radius=10, color=(0, 0, 255),
                       thickness=-1)

        # if frame_idx == 138:
        #     cv2.imshow('test', rgb_img0)
        #     if cv2.waitKey(0) & 0xFF == 27:
        #         break
        #     print()
        # cv2.imshow('test', rgb_img0)
        # cv2.waitKey(100)
        # if cv2.waitKey(0) & 0xFF == 27:
        #     break

    print('done')


if __name__ == '__main__':
    run()
