import os

import numpy as np
from torch.utils.data import Dataset
import cv2 as cv
import json
import shutil
import matplotlib.pyplot as plt
from module.Center_HM_labels import get_HM_label, get_OneStage_lidar_annotation

class self_dataset(Dataset):
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        # self.common_info_root = os.path.join(dataset_root, 'common_info')
        self.scene = os.listdir(dataset_root)   # every frame path
        # self.scene.remove('common_info')
        self.scene.sort()
        self.rgb_data = []
        self.radar_data = []
        self.lidar_data = []
        self.scene_path = []
        for scene in self.scene:
            # ScenePath = os.path.join(os.path.join(dataset_root, scene))
            self.scene_path.append(scene)

            ti_path = os.path.join(os.path.join(dataset_root, scene, 'TIRadar'))
            # dynamic_heatmap_path = ''
            # static_heatmap_path = ''
            for file in os.listdir(ti_path):
                if file[0] == 'd':
                    dynamic_heatmap_path = os.path.join(ti_path, file)
                if file[0] == 's':
                    static_heatmap_path = os.path.join(ti_path, file)
            heatmap_path = [dynamic_heatmap_path, static_heatmap_path]
            # print('heatmap_path', heatmap_path)
            self.radar_data.append(heatmap_path)

            lidar_path = os.path.join(os.path.join(dataset_root, scene, 'VelodyneLidar'))
            for file in os.listdir(lidar_path):
                if file[-4:] == 'json':
                    lidar_json_path = os.path.join(lidar_path, file)
            self.lidar_data.append(lidar_json_path)

            camera_path = os.path.join(dataset_root, scene, 'LeopardCamera1')
            for file in os.listdir(camera_path):
                if file[-3:] == 'png':
                    rgb_path = os.path.join(camera_path, file)
            self.rgb_data.append(rgb_path)

        assert len(self.radar_data) == len(self.lidar_data)
        assert len(self.rgb_data) == len(self.lidar_data)
        self.length = len(self.radar_data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        with open(self.lidar_data[item]) as f:
            lidar_json_data = json.load(f)
        target_num = len(lidar_json_data['annotation'])
        ScenePath = self.scene_path[item]

        # rgb_img = cv.imread(self.rgb_data[item])
        # dynamic_heatmap = cv.imread(self.radar_data[item][0])
        # static_heatmap = cv.imread(self.radar_data[item][1])
        # # print(self.radar_data[item][0], '\n', self.radar_data[item][1])
        # # with open(self.lidar_data[item]) as f:
        # #     lidar_json = json.load(f)
        # # lidar_annotation = lidar_json['annotation']
        # HM_x, HM_y, HM_l, HM_w, HM_alpha = get_OneStage_lidar_annotation(self.lidar_data[item])
        # # print('HM_x:', len(HM_x))
        # # print('HM_y:', HM_y)
        # One_Stage_HM = get_HM_label(HM_x, HM_y, HM_l, HM_w, HM_alpha)
        #
        # rgb_img = cv.resize(rgb_img, (1024, 1024))
        # # rgb_img = rgb_img[:, 500:500+2401, :]
        # dynamic_height = dynamic_heatmap.shape[0] // 3
        # static_height = dynamic_heatmap.shape[0] // 3
        # dynamic_width = dynamic_heatmap.shape[1] // 2
        # dynamic_width_25m = dynamic_width // 3
        # static_width = static_heatmap.shape[1] // 2
        # static_width_25m = static_width // 3
        # dynamic_heatmap = dynamic_heatmap[dynamic_height:, dynamic_width-dynamic_width_25m:dynamic_width+dynamic_width_25m, :]
        # static_heatmap = static_heatmap[static_height:, static_width-static_width_25m:static_width+static_width_25m, :]
        # dynamic_heatmap = cv.resize(dynamic_heatmap, (256, 256))
        # static_heatmap = cv.resize(static_heatmap, (256, 256))
        #
        # return One_Stage_HM, rgb_img, dynamic_heatmap, static_heatmap

        return target_num, ScenePath


if __name__ == '__main__':
    dataset_train = self_dataset('/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train')

    dataset_len = dataset_train.length
    old_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
    new_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train2'

    for i in range(dataset_len):
        TargetNum, ScenePath = dataset_train[i]
        if TargetNum >= 8:
            src = os.path.join(old_path, ScenePath)
            dst = os.path.join(new_path, ScenePath)
            shutil.move(src, dst)

    # TargetNum, ScenePath = dataset_train[0]
    # print('ScenePath:', ScenePath)

    # One_Stage_HM, rgb_img, dynamic_heatmap, static_heatmap = dataset_train[0]
    # print('dynamic_heatmap', dynamic_heatmap.shape)
    # print('static_heatmap', static_heatmap.shape)
    #
    # # print('One_Stage_HM:', One_Stage_HM)
    # plt.pcolor(One_Stage_HM[0])
    # plt.axis('off')
    # plt.show()
    #
    # # rgb_img = cv.resize(rgb_img, (1024, 1024))
    # # cv.imshow('rgb_img', rgb_img)
    # cv.imshow('dynamic_heatmap', dynamic_heatmap)
    # cv.imshow('static_heatmap', static_heatmap)
    # cv.waitKey(0)

    print('done')

