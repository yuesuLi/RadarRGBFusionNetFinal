import os
import re
import sys
sys.path.append('../Detection/yolov5')
print("*****************", sys.path)
import time
import math
import json
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
# from Detection.detection import Detection

from TiProcess.BEV_TI import read_pcd, load_json, pts2rbev
from TiProcess.proj_radar2cam import cam_to_radar, cam_to_radar2


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

        frame_num += 1
        TI_points = read_pcd(TI_pcd_path)  # (N, 3)
        lidar_json = load_json(calib_lidar)
        TI_json = load_json(TI_json_path)
        cam_json = load_json(img_json)
        TI_timestamp = TI_json['timestamp']

        # lidar2TI_anno = get_lidar2TI_anno(lidar_json, TI_json)
        print('TI_pcd_path:', TI_pcd_path)
        print('img_path:', img_path)
        Lidar_annotation_points = pts2rbev(lidar_json, TI_json, cam_json)
        print('TI_timestamp:', TI_timestamp)
        print('TI_points: ', TI_points.shape, '\n')

        mask = np.ones(TI_points.shape[0], dtype=bool)
        mask = np.logical_and(mask, TI_points[:, 1] <= 40)
        mask = np.logical_and(mask, TI_points[:, 0] <= 20)
        mask = np.logical_and(mask, TI_points[:, 0] >= -10)
        TI_points = TI_points[mask, :]
        TI_points = TI_points[:, 0:2]

        TI_db = DBSCAN(eps=0.5, min_samples=5).fit(TI_points)
        TI_dbscan_points = TI_points[TI_db.labels_[:] != -1]
        print('TI_dbscan_points: ', TI_dbscan_points.shape)
        TI_num_dbscan = len(np.unique(TI_db.labels_)) - (1 if -1 in TI_db.labels_ else 0)
        print("TI_num_dbscan: ", TI_num_dbscan)
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
                z = 1.7
                x = (bbox_15[:, 0] + bbox_15[:, 2]) / 2
                y = np.minimum(bbox_15[:, 1], bbox_15[:, 3])

                # x = bbox_15[:, 0] + bbox_15[:, 2] / 2
                # y = bbox_15[:, 1] + bbox_15[:, 3]

                # cam_to_radar_point = cam_to_radar(x, y, -z, RT1, V1)
                cam_to_radar_point = cam_to_radar2(bbox_15, z, RT1, V1)
                for i in range(len(cam_to_radar_point)):
                    x = cam_to_radar_point[i][0]
                    y = cam_to_radar_point[i][1]
                    plt.scatter(x, y, s=50, marker='s', c='r')

            if folder == '20211025_1_group0012_frame0020_labeled':
                RT1 = np.array(TI_json['TIRadar_to_LeopardCamera1_TransformMatrix'])
                R1 = RT1[:, 0:3]
                T1 = RT1[:, 3:4]
                V1 = np.array(cam_json['IntrinsicMatrix'])
                z = 1.7
                x = (bbox_20[:, 0] + bbox_20[:, 2]) / 2
                y = np.minimum(bbox_20[:, 1], bbox_20[:, 3])

                # cam_to_radar_point = cam_to_radar(x, y, z, R1, T1, V1)
                cam_to_radar_point = cam_to_radar2(bbox_20, z, RT1, V1)
                for i in range(len(cam_to_radar_point)):
                    x = cam_to_radar_point[i][0]
                    y = cam_to_radar_point[i][1]
                    plt.scatter(x, y, s=50, marker='s', c='r')

            color_list = plt.cm.tab20(np.linspace(0, 1, TI_num_dbscan))
            for i in range(TI_num_dbscan):
                points_class = TI_points[TI_db.labels_ == i, :]
                # color = cmap[int(np.floor(255/num_clusters) * db.labels_[i]), :]/255
                color = color_list[i]
                plt.scatter(points_class[:, 0], points_class[:, 1], 5, color=tuple(color))
            for i in range(len(Lidar_annotation_points)):
                x = Lidar_annotation_points[i][0]
                y = Lidar_annotation_points[i][1]
                plt.scatter(x, y, s=50, marker='x', c='k')

            plt.show()
            plt.pause(0.1)
            plt.clf()


if __name__ == '__main__':
    run()
    print('done')
