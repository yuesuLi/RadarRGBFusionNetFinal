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


class self_datasetExtend(Dataset):
    def __init__(self, dataset_root, anno_root, img_size=[640]):
        self.dataset_root = dataset_root    # data group path
        self.anno_root = anno_root          # anno group path
        self.img_size = [640]
        self.stride = 32
        self.auto = False
        self.img_size *= 2 if len(self.img_size) == 1 else 1  # expand


        # self.common_info_root = os.path.join(dataset_root, 'common_info')
        self.scenes = os.listdir(self.anno_root)      # frames path
        # self.scenes = os.listdir(self.dataset_root)
        # self.scenes.remove('common_info')
        self.scenes.sort()
        self.rgb_data = []
        self.rgb_jsons = []
        self.OCU_data = []
        self.OCU_jsons = []
        self.lidar_data = []
        self.lidar_jsons = []
        self.extend_annos = []

        for currScene in self.scenes:
            curr_extend_anno = os.path.join(self.anno_root, currScene, 'extendGT.json')
            self.extend_annos.append(curr_extend_anno)

            scene = os.path.join(self.dataset_root, currScene)
            ocu_path = os.path.join(scene, 'OCULiiRadar')
            for file in os.listdir(ocu_path):
                if file[-3:] == 'pcd':
                    OCU_points_path = os.path.join(ocu_path, file)
                if file[-4:] == 'json':
                    OCU_json_path = os.path.join(ocu_path, file)
            self.OCU_data.append(OCU_points_path)
            self.OCU_jsons.append(OCU_json_path)

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

        assert len(self.lidar_jsons) == len(self.extend_annos)
        assert len(self.OCU_data) == len(self.extend_annos)
        assert len(self.rgb_data) == len(self.extend_annos)
        self.length = len(self.extend_annos)

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        extendGT = load_json(self.extend_annos[item])
        # print('Img_Path:', self.rgb_data[item])
        rgb_json = load_json(self.rgb_jsons[item])
        rgb_img = cv2.imread(self.rgb_data[item])

        rgb_img = undistort_image(rgb_img, rgb_json)
        # rgb_img = rgb_img[:, 225:736, :]

        # cv2.imshow('rgb_img', rgb_img)
        # cv2.waitKey(0)
        img, ratio, pad = letterbox(rgb_img, self.img_size, stride=self.stride, auto=self.auto)
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


        return rgb_img, img, rgb_json, OCU_data, OCU_json, lidar_json, extendGT


if __name__ == '__main__':
    data_base_path = '/mnt/ourDataset_v2/ourDataset_v2'
    anno_base_path = '/mnt/ChillDisk/personal_data/zhangq/RadarRGBFusionNet2_20231128/dataset/ExtendGT'
    group_path = '20221220_group0018_mode2_461frames'

    source = os.path.join(data_base_path, group_path)
    anno_group_path = os.path.join(anno_base_path, group_path)
    dataset_train = self_datasetExtend(source, anno_group_path)
    rgb_img, img, rgb_json, OCU_data, OCU_json, lidar_json, extendGT = dataset_train[0]
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False,
                                                   num_workers=1, drop_last=True)

    for frameID, (rgb_img, img, rgb_json, OCU_data, OCU_json, lidar_json, extendGT) in enumerate(dataset_train):

        print('frameID', frameID)
        print(extendGT)



    print('done')

    print('TI_data', OCU_data.shape)

    cv2.imshow('rgb_img', rgb_img)
    cv2.waitKey(0)
    # if cv2.waitKey(0) & 0xFF == 27:
    #     break

    print('done')

