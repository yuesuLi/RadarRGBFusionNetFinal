import os
import re
import sys
sys.path.append('../Detection/yolov5')
sys.path.append('../Detection/deep/reid')
# print("******************", sys.path)

import time
import math
import json
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib
# print(matplotlib.get_backend())
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from Detection.detection import Detection

from TiProcess.BEV_TI import read_pcd, load_json, pts2rbev
from TiProcess.proj_radar2cam import cam_to_radar, cam_to_radar2


def get_dbscan_points(raw_points, eps, min_samples):

    length = raw_points.shape[0]
    points = np.concatenate([raw_points[:, 0].reshape((length, 1)),
                             raw_points[:, 2].reshape((length, 1)),
                             raw_points[:, 1].reshape((length, 1)),
                             raw_points[:, 3].reshape((length, 1)),
                             raw_points[:, 4].reshape((length, 1))], axis=1)

    # mask = np.ones(points.shape[0], dtype=bool)
    # mask = np.logical_and(mask, points[:, 1] <= 50)
    # mask = np.logical_and(mask, points[:, 1] >= 0)
    # mask = np.logical_and(mask, points[:, 0] <= 10)
    # mask = np.logical_and(mask, points[:, 0] >= -10)
    # points = points[mask, :]
    # points = points[:, 0:2]

    if points.shape[0] == 0:
        return points, None, 0

    points_db = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, 0:2])
    # points = points[points_db.labels_[:] != -1]
    # print('TI_dbscan_points: ', points.shape)
    num_cluster = len(np.unique(points_db.labels_)) - (1 if -1 in points_db.labels_ else 0)
    # print("TI_num_dbscan: ", num_cluster)

    return points, points_db, num_cluster

# points: (n, 5)   x y z vel SNR
def get_radar_features(points, points_db, num_cluster):

    # cluster_points = {}
    r_detection = []
    if num_cluster == 0:
        return r_detection
    # points_features = []
    labels = points_db.labels_
    for i in range(num_cluster):
        one_cluster = points[labels == i, :]

        r_center = np.mean(one_cluster[:, :2], axis=0)

        if r_center[0] > 10 or r_center[0] < -10 or r_center[1] > 50 or r_center[1] < 0:
            continue

        a1 = one_cluster.shape[0]
        a2 = max(one_cluster[:, 0]) - min(one_cluster[:, 0])
        a3 = max(one_cluster[:, 1]) - min(one_cluster[:, 1])
        a4 = a2 * a3
        a5 = a1 / (a4+1e-6)
        a6 = max(one_cluster[:, 2]) - min(one_cluster[:, 2])
        a7 = np.mean(one_cluster[:, 2])
        a8 = np.mean(one_cluster[:, 3])
        a9 = max(one_cluster[:, 3]) - min(one_cluster[:, 3])
        a10 = max(one_cluster[:, 4])
        radar_feature = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])
        # points_features.append([a1, a2, a3, a4, a5, a6, a7, a8, a9])
        r_detection.append(Detection(center=r_center, fusion_state=2, r_center=r_center, r_feature=radar_feature))

        # color = cmap[int(np.floor(255/num_cluster) * labels[i]), :]/255
        # plt.scatter(cluster_points[i][:, 0], cluster_points[i][:, 1], 5, color=tuple(color))

    # print(' ')
    return r_detection





def run():
    base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'

    folders = os.listdir(base_path)
    folders = sorted(folders)
    frame_num = -1
    plot = True
    for folder in folders:
        if 'labeled' not in folder:
            continue
        camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
        for file in os.listdir(camera_path):
            if file[-3:] == 'png':
                img_path = os.path.join(camera_path, file)
            if file[-4:] == 'json':
                img_json = os.path.join(camera_path, file)
        lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
        for file in os.listdir(lidar_path):
            if file[-3:] == 'pcd':
                pcd_lidar = os.path.join(lidar_path, file)
            if file[-4:] == 'json':
                calib_lidar = os.path.join(lidar_path, file)
        ti_path = os.path.join(base_path, folder, 'TIRadar')
        for file in os.listdir(ti_path):
            if file[-3:] == 'pcd':
                TI_pcd_path = os.path.join(ti_path, file)
            if file[-4:] == 'json':
                TI_json_path = os.path.join(ti_path, file)
        ocu_path = os.path.join(base_path, folder, 'OCULiiRadar')
        for file in os.listdir(ocu_path):
            if file[-3:] == 'pcd':
                OCU_pcd_path = os.path.join(ocu_path, file)
            if file[-4:] == 'json':
                OCU_json_path = os.path.join(ocu_path, file)

        frame_num += 1
        TI_points = read_pcd(TI_pcd_path)  # (N, 3)
        lidar_json = load_json(calib_lidar)
        TI_json = load_json(TI_json_path)
        cam_json = load_json(img_json)
        TI_timestamp = TI_json['timestamp']
        Lidar_annotation_points = pts2rbev(lidar_json, TI_json, cam_json)

        # lidar2TI_anno = get_lidar2TI_anno(lidar_json, TI_json)
        print('img_path:', img_path)
        # print('TI_pcd_path:', TI_pcd_path)
        # print('TI_timestamp:', TI_timestamp)
        # print('TI_points: ', TI_points.shape, '\n')

        # TI_points: (n, 5)   x y z vel SNR
        TI_points, TI_db, TI_num_dbscan = get_dbscan_points(TI_points, 0.9, 7)
        r_detections = get_radar_features(TI_points, TI_db, TI_num_dbscan)

        radar2img_matrix = np.array(TI_json['TIRadar_to_LeopardCamera1_TransformMatrix'])
        IntrinsicMatrix = np.array(cam_json['IntrinsicMatrix'])
        RT_Matrix = np.array(np.dot(np.linalg.inv(IntrinsicMatrix), radar2img_matrix))
        # print('RT_Matrix', RT_Matrix)
        R = RT_Matrix[:, 0:3]
        # print('R', R)
        theta_x = np.arctan2(R[2, 1], R[2, 2])
        theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 0]**2 + R[2, 2]**2))
        theta_z = np.arctan2(R[1, 0], R[0, 0])
        # print('theta:', theta_x, theta_y, theta_z)
        # plot = False



        # TI_colors = np.random.rand(TI_num_dbscan, 3)
        bbox_15 = np.array([[1959.5, 1016.2, 189.5, 132.4, 2], [2290, 1030, 316.2, 170.7, 2]])
        bbox_20 = np.array([[1949.6, 1016.2, 170.7, 123.5, 2], [2241, 1030, 290.5, 159.79, 2], [852.2, 1002.3, 248.39, 128.79, 2],
                            [1005.4, 1018.4, 243.8, 139.39, 2], [3343.4, 1549, 173.59, 151, 2]])

        if plot:

            # if TI_num_dbscan == 0:
            plt.subplot(2, 1, 1)
            plt.title(folder)
            plt.xlim(-20, 20)
            plt.ylim(0, 40)
            plt.scatter(TI_points[:, 0], TI_points[:, 1], 5)
            for i in range(len(Lidar_annotation_points)):
                x = Lidar_annotation_points[i][0]
                y = Lidar_annotation_points[i][1]
                plt.scatter(x, y, s=50, marker='x', c='k')
            # else:
            plt.subplot(2, 1, 2)
            plt.xlim(-20, 20)
            plt.ylim(0, 40)

            if folder == '20211025_1_group0012_frame0015_labeled':
                RT1 = np.array(TI_json['TIRadar_to_LeopardCamera1_TransformMatrix'])
                R1 = RT1[:, 0:3]
                T1 = RT1[:, 3:4]
                V1 = np.array(cam_json['IntrinsicMatrix'])
                x = (bbox_15[:, 0] + bbox_15[:, 2]) / 2
                y = np.minimum(bbox_15[:, 1], bbox_15[:, 3])

                # x = bbox_15[:, 0] + bbox_15[:, 2] / 2
                # y = bbox_15[:, 1] + bbox_15[:, 3]

                # cam_to_radar_point = cam_to_radar(x, y, -z, RT1, V1)
                cam_to_radar_point = cam_to_radar2(bbox_15, RT1, V1)
                for i in range(len(cam_to_radar_point)):
                    x = cam_to_radar_point[i][0]
                    y = cam_to_radar_point[i][1]
                    plt.scatter(x, y, s=50, marker='s', c='r')

            if folder == '20211025_1_group0012_frame0020_labeled':
                RT1 = np.array(TI_json['TIRadar_to_LeopardCamera1_TransformMatrix'])
                R1 = RT1[:, 0:3]
                T1 = RT1[:, 3:4]
                V1 = np.array(cam_json['IntrinsicMatrix'])
                x = (bbox_20[:, 0] + bbox_20[:, 2]) / 2
                y = np.minimum(bbox_20[:, 1], bbox_20[:, 3])

                # cam_to_radar_point = cam_to_radar(x, y, z, R1, T1, V1)
                cam_to_radar_point = cam_to_radar2(bbox_20, RT1, V1)
                for i in range(len(cam_to_radar_point)):
                    x = cam_to_radar_point[i][0]
                    y = cam_to_radar_point[i][1]
                    plt.scatter(x, y, s=50, marker='s', c='r')

            color_list = plt.cm.tab20(np.linspace(0, 1, TI_num_dbscan))
            for i in range(TI_num_dbscan):
                points_class = TI_points[TI_db.labels_ == i, :]
                # color = cmap[int(np.floor(255/num_clusters) * db.labels_[i]), :]/255
                color = color_list[i]
                plt.scatter(points_class[:, 0], points_class[:, 1], s=5, color=tuple(color))
            for i in range(len(Lidar_annotation_points)):
                x = Lidar_annotation_points[i][0]
                y = Lidar_annotation_points[i][1]
                plt.scatter(x, y, s=50, marker='x', c='k')

            plt.show()
            plt.pause(0.1)
            # save_name = '/media/personal_data/zhangq/RadarRGBFusionNet/results/' + folder + '.png'
            # plt.savefig(save_name, dpi=500)
            plt.clf()


if __name__ == '__main__':
    run()
    print('done')
