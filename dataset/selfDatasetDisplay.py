import os

import numpy as np
from dataset.get_radar_points import read_pcd
from torch.utils.data import Dataset
import cv2
import json
import torch
import matplotlib.pyplot as plt
from module.Center_HM_labels import get_HM_label, get_OneStage_lidar_annotation


def undistort_image(image, rgb_json):
    intrinsic = rgb_json['intrinsic']
    radial_distortion = rgb_json['radial_distortion']
    tangential_distortion = rgb_json['tangential_distortion']
    k1, k2, k3 = radial_distortion
    p1, p2 = tangential_distortion

    image_undistort = cv2.undistort(image, np.array(intrinsic), np.array([k1, k2, p1, p2, k3]))
    # all_img = np.hstack((image, image_undistort))

    return image_undistort

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


class self_dataset3(Dataset):
    def __init__(self, dataset_root, img_size=[640]):
        self.dataset_root = dataset_root
        self.img_size = [640]
        self.stride = 32
        self.auto = False
        self.img_size *= 2 if len(self.img_size) == 1 else 1  # expand


        # self.common_info_root = os.path.join(dataset_root, 'common_info')
        self.scenes = os.listdir(dataset_root)   # every frame path
        # self.scenes.remove('common_info')
        self.scenes.sort()
        self.rgb_data = []
        self.rgb_jsons = []
        self.TI_data = []
        self.TI_jsons = []
        self.OCU_data = []
        self.OCU_jsons = []
        self.lidar_data = []
        self.lidar_jsons = []

        for currScene in self.scenes:
            scene = os.path.join(dataset_root, currScene)
            ocu_path = os.path.join(scene, 'OCULiiRadar')
            for file in os.listdir(ocu_path):
                if file[-3:] == 'pcd':
                    OCU_points_path = os.path.join(ocu_path, file)
                if file[-4:] == 'json':
                    OCU_json_path = os.path.join(ocu_path, file)
            self.OCU_data.append(OCU_points_path)
            self.OCU_jsons.append(OCU_json_path)

            # ti_path = os.path.join(scene, 'TIRadar')
            # for file in os.listdir(ti_path):
            #     if file[-3:] == 'pcd':
            #         OCU_points_path = os.path.join(ti_path, file)
            #     if file[-4:] == 'json':
            #         OCU_json_path = os.path.join(ti_path, file)
            # self.OCU_data.append(OCU_points_path)
            # self.OCU_jsons.append(OCU_json_path)

            lidar_path = os.path.join(os.path.join(scene, 'VelodyneLidar'))
            for file in os.listdir(lidar_path):
                if file[-4:] == 'json':
                    lidar_json_path = os.path.join(lidar_path, file)
                if file[-3:] == 'pcd':
                    Lidar_points_path = os.path.join(lidar_path, file)
            self.lidar_jsons.append(lidar_json_path)
            self.lidar_data.append(Lidar_points_path)

            camera_path = os.path.join(dataset_root, scene, 'LeopardCamera0')
            for file in os.listdir(camera_path):
                if file[-3:] == 'png':
                    rgb_path = os.path.join(camera_path, file)
                if file[-4:] == 'json':
                    rgb_json_path = os.path.join(camera_path, file)
            self.rgb_data.append(rgb_path)
            self.rgb_jsons.append(rgb_json_path)

        # for scene in self.scenes:
        #     # if 'labeled' not in scene:
        #     #     continue
        #     # true_dirs = os.path.join(os.readlink(soft_link), 'LeopardCamera0')
        #     # soft_link = os.path.join(dataset_root, scene)
        #     # scene = os.readlink(soft_link)
        #     scene = os.path.join(dataset_root, scene)



        assert len(self.OCU_data) == len(self.lidar_jsons)
        assert len(self.rgb_data) == len(self.lidar_jsons)
        self.length = len(self.lidar_jsons)

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        # print('Img_Path:', self.rgb_data[item])
        rgb_json = load_json(self.rgb_jsons[item])
        rawImg = cv2.imread(self.rgb_data[item])

        rawImg = undistort_image(rawImg, rgb_json)
        rawImg = rawImg[:, 225:736, :]

        # cv2.imshow('rawImg', rawImg)
        # cv2.waitKey(0)
        img, ratio, pad = letterbox(rawImg, self.img_size, stride=self.stride, auto=self.auto)
        self.img_size *= 2 if len(self.img_size) == 1 else 1  # expand

        # Stack
        img = np.stack(img, 0)
        # Convert
        img = img.transpose((2, 0, 1))
        # img = img[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        # TI_data = read_pcd(self.TI_data[item])      # x y z vel SNR
        # TI_json = load_json(self.TI_jsons[item])
        OCU_data = read_pcd(self.OCU_data[item])  # x y z doppler snr
        OCU_json = load_json(self.OCU_jsons[item])

        # Lidar_points = read_pcd(self.lidar_data[item])  # x y z intensity idx_laser
        lidar_json = load_json(self.lidar_jsons[item])
        # curr_label = load_json(self.lidar_jsons[item])
        # curr_label = curr_label['annotation']
        # lidar_annotation = load_json(self.lidar_jsons[item])['annotation']
        # with open(self.lidar_jsons[item]) as f:
        #     lidar_json = json.load(f)
        # lidar_annotation = lidar_json['annotation']
        Lidar_points = []

        return rawImg, img, rgb_json, OCU_data, OCU_json, lidar_json, Lidar_points


if __name__ == '__main__':

    import openpyxl

    file_path = '/mnt/ChillDisk/personal_data/zhangq/RadarRGBFusionNet2_20231128/GroupPath/Display.xlsx'
    DataPath = openpyxl.load_workbook(file_path)
    ws = DataPath.active
    groups_excel = ws['A']
    datasets_path = []
    for cell in groups_excel:
        datasets_path.append(cell.value)

    data_base_path = '/mnt/ourDataset_v2/ourDataset_v2'
    source = os.path.join(data_base_path, datasets_path[0])
    dataset_train = self_dataset3(source)
    for frameID, (rawImg, img, rgb_json, OCU_data, OCU_json, lidar_json, Lidar_points) in enumerate(dataset_train):
        print('frameID', frameID)
        # print('OCU_data', OCU_data.shape)
        if 'annotation' in lidar_json:
            print('lidar_json', lidar_json)
        # lidarJsonFile = load_json(lidar_json)

        # cv2.imshow('rawImg', rawImg)
        # cv2.waitKey(10)
    # if cv2.waitKey(0) & 0xFF == 27:
    #     break
    print('numFrames:', frameID)
    print('done')

