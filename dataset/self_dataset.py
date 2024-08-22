import os

import numpy as np
from dataset.get_radar_points import read_pcd
from torch.utils.data import Dataset
import cv2
import json
import torch
import matplotlib.pyplot as plt
from module.Center_HM_labels import get_HM_label, get_OneStage_lidar_annotation




def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('im', im)
        # cv2.waitKey(0)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # cv2.imshow('im', im)
    # cv2.waitKey(0)
    return im, ratio, (dw, dh)

def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data


class self_dataset(Dataset):
    def __init__(self, dataset_root, img_size=[640]):
        self.dataset_root = dataset_root
        self.img_size = [640]
        self.stride = 32
        self.auto = False
        self.img_size *= 2 if len(self.img_size) == 1 else 1  # expand


        # self.common_info_root = os.path.join(dataset_root, 'common_info')
        self.scene = os.listdir(dataset_root)   # every frame path
        # self.scene.remove('common_info')
        self.scene.sort()
        self.rgb_data = []
        self.rgb_jsons = []
        self.TI_data = []
        self.TI_jsons = []
        self.lidar_data = []
        for scene in self.scene:
            if 'labeled' not in scene:
                continue
            ti_path = os.path.join(os.path.join(dataset_root, scene, 'TIRadar'))
            for file in os.listdir(ti_path):
                if file[-3:] == 'pcd':
                    TI_points_path = os.path.join(ti_path, file)
                if file[-4:] == 'json':
                    TI_json_path = os.path.join(ti_path, file)
            self.TI_data.append(TI_points_path)
            self.TI_jsons.append(TI_json_path)

            lidar_path = os.path.join(os.path.join(dataset_root, scene, 'VelodyneLidar'))
            for file in os.listdir(lidar_path):
                if file[-4:] == 'json':
                    lidar_json_path = os.path.join(lidar_path, file)
            self.lidar_data.append(lidar_json_path)

            camera_path = os.path.join(dataset_root, scene, 'LeopardCamera1')
            for file in os.listdir(camera_path):
                if file[-3:] == 'png':
                    rgb_path = os.path.join(camera_path, file)
                if file[-4:] == 'json':
                    rgb_json_path = os.path.join(camera_path, file)
            self.rgb_data.append(rgb_path)
            self.rgb_jsons.append(rgb_json_path)


        assert len(self.TI_data) == len(self.lidar_data)
        assert len(self.rgb_data) == len(self.lidar_data)
        self.length = len(self.lidar_data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        # print('Img_Path:', self.rgb_data[item])
        rgb_json = load_json(self.rgb_jsons[item])
        rgb_img = cv2.imread(self.rgb_data[item])
        img, ratio, pad = letterbox(rgb_img, self.img_size, stride=self.stride, auto=self.auto)
        self.img_size *= 2 if len(self.img_size) == 1 else 1  # expand

        # Stack
        img = np.stack(img, 0)
        # Convert
        img = img.transpose((2, 0, 1))
        # img = img[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        TI_data = read_pcd(self.TI_data[item])      # x y z vel SNR
        TI_json = load_json(self.TI_jsons[item])

        lidar_json = load_json(self.lidar_data[item])
        # lidar_annotation = load_json(self.lidar_data[item])['annotation']
        # with open(self.lidar_data[item]) as f:
        #     lidar_json = json.load(f)
        # lidar_annotation = lidar_json['annotation']

        return rgb_img, img, rgb_json, TI_data, TI_json, lidar_json


if __name__ == '__main__':
    dataset_train = self_dataset('/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled')
    rgb_img, img, rgb_json, TI_data, TI_json, lidar_annotation = dataset_train[0]
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False,
                                                   num_workers=1, drop_last=True)

    # for step, (rgb_img0, img, TI_data, lidar_annotation) in enumerate(dataloader_train):
    #     rgb_img0 = rgb_img0.float()

    print('TI_data', TI_data.shape)

    cv2.imshow('rgb_img', rgb_img)
    cv2.waitKey(0)
    # if cv2.waitKey(0) & 0xFF == 27:
    #     break

    print('done')

