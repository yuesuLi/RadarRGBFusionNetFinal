import argparse
import os
import sys
from pathlib import Path
import numpy as np
import json

import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm

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

def read_pcd(pcd_path):
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
                             pc_data[metadata['fields'][1]][:, None],
                             pc_data[metadata['fields'][2]][:, None],
                             pc_data[metadata['fields'][3]][:, None],
                             pc_data[metadata['fields'][4]][:, None]], axis=-1)
    # print(points.shape)
    return points

def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data


def pointcloud_transform(pointcloud, transform_matrix):
    '''
        transform pointcloud from coordinate1 to coordinate2 according to transform_matrix
    :param pointcloud: (x, y, z, id, classes, ...) N*5
    :param transform_matrix: 4*4
    :return pointcloud_transformed: (x, y, z, ...)
    '''
    n_points = pointcloud.shape[0]
    xyz = pointcloud[:, :3]
    xyz1 = np.vstack((xyz.T, np.ones((1, n_points))))
    xyz1_transformed = np.matmul(transform_matrix, xyz1)
    pointcloud_transformed = np.hstack((
        xyz1_transformed[:3, :].T,
        pointcloud[:, 3:]
    ))
    return pointcloud_transformed

def pts2rbev(lidar_json, TI_json, cam_json):
    # classes 2 means car
    # Get LiDAR annotation points
    lidar_raw_anno = lidar_json['annotation']
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
        gt_id.append(int(lidar_raw_anno[idx]['id']))
        gt_classes.append(2)

    if len(x) > 0:
        # Lidar_anno_points = np.vstack((x, y, z)).T   # (N, 3)
        Lidar_anno_points = np.vstack((x, y, z, gt_id, gt_classes)).T  # (N, 5)
    else:
        Lidar_anno_points = np.array([])
        return Lidar_anno_points


    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera1_TransformMatrix = np.array(lidar_json['VelodyneLidar_to_LeopardCamera1_TransformMatrix'])
    TIRadar_to_LeopardCamera1_TransformMatrix = np.array(TI_json['TIRadar_to_LeopardCamera1_TransformMatrix'])
    # TIRadar_to_LeopardCamera1_TransformMatrix = np.array(TI_json['OCULiiRadar_to_LeopardCamera1_TransformMatrix'])
    LeopardCamera1_IntrinsicMatrix = np.array(cam_json['IntrinsicMatrix'])
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )

    Lidar_anno_points = pointcloud_transform(Lidar_anno_points, VelodyneLidar_to_TIRadar_TransformMatrix)

    # return Lidar_anno_points

    final_x, final_y, final_z, final_id, final_classes = [], [], [], [], []
    for i in range(len(Lidar_anno_points)):
        tmp_x = Lidar_anno_points[i][0]
        tmp_y = Lidar_anno_points[i][1]
        if tmp_x >= 20 or tmp_x <= -10 or tmp_y >= 40 or tmp_y <= 0:
            continue
        final_x.append(tmp_x)
        final_y.append(tmp_y)
        final_z.append(Lidar_anno_points[i][2])
        final_id.append(Lidar_anno_points[i][3])
        final_classes.append(Lidar_anno_points[i][4])

    if len(final_x) > 0:
        Lidar_dinal_anno_points = np.vstack((final_id, final_x, final_y, final_z, final_classes)).T   # (N, 5)
    else:
        Lidar_dinal_anno_points = np.array([])

    return Lidar_dinal_anno_points

def OCUpts2rbev(lidar_json, OCU_json, cam_json, curr_label):
    # classes 2 means car
    # Get LiDAR annotation points
    # lidar_raw_anno = load_json(curr_label)
    lidar_raw_anno = curr_label
    # lidar_raw_anno = lidar_json['annotation']
    x, y, z, gt_id, gt_classes = [], [], [], [], []
    for idx in range(len(lidar_raw_anno)):
        if lidar_raw_anno[idx]['class'] != 'car':
            continue
        x.append(lidar_raw_anno[idx]['x'])
        y.append(lidar_raw_anno[idx]['y'])
        z.append(lidar_raw_anno[idx]['z'])
        gt_id.append(int(lidar_raw_anno[idx]['object_id']))
        gt_classes.append(2)

    if len(x) > 0:
        # Lidar_anno_points = np.vstack((x, y, z)).T   # (N, 3)
        Lidar_anno_points = np.vstack((x, y, z, gt_id, gt_classes)).T  # (N, 5)
    else:
        Lidar_anno_points = np.array([])
        return Lidar_anno_points


    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera0_TransformMatrix = np.array(lidar_json['VelodyneLidar_to_LeopardCamera0_extrinsic']) # 4*4
    OCU_to_LeopardCamera1_TransformMatrix = np.array(OCU_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])   # 4*4
    # TIRadar_to_LeopardCamera1_TransformMatrix = np.array(TI_json['OCULiiRadar_to_LeopardCamera1_TransformMatrix'])
    # LeopardCamera1_IntrinsicMatrix = np.array(cam_json['intrinsic'])
    VelodyneLidar2OCU_TransformMatrix = np.matmul(np.linalg.inv(OCU_to_LeopardCamera1_TransformMatrix),
        VelodyneLidar_to_LeopardCamera0_TransformMatrix
    )

    Lidar_anno_points = pointcloud_transform(Lidar_anno_points, VelodyneLidar2OCU_TransformMatrix)

    # return Lidar_anno_points

    final_x, final_y, final_z, final_id, final_classes = [], [], [], [], []
    for i in range(len(Lidar_anno_points)):
        tmp_x = Lidar_anno_points[i][0]
        tmp_y = Lidar_anno_points[i][1]
        tmp_z = Lidar_anno_points[i][2]

        if tmp_x >= 10 or tmp_x <= -10 or tmp_z >= 40 or tmp_z <= 5:
            continue
        final_x.append(tmp_x)
        final_y.append(tmp_z)
        final_z.append(tmp_y)
        final_id.append(Lidar_anno_points[i][3])
        final_classes.append(Lidar_anno_points[i][4])

    if len(final_x) > 0:
        Lidar_final_anno_points = np.vstack((final_id, final_x, final_y, final_z, final_classes)).T   # (N, 5)
    else:
        Lidar_final_anno_points = np.array([])

    return Lidar_final_anno_points


def get_lidar2TI_anno(lidar_json, TI_json):
    lidar_raw_anno = lidar_json['annotation']
    lidar2cam_TransformMatrix  = lidar_json['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    TI2cam_TransformMatrix = TI_json['TIRadar_to_LeopardCamera1_TransformMatrix']

    R_Lidar = lidar2cam_TransformMatrix[:, 0:3]
    T_Lidar = lidar2cam_TransformMatrix[:, 3:4]
    R_TI = TI2cam_TransformMatrix[:, 0:3]
    T_TI = TI2cam_TransformMatrix[:, 3:4]

    lidar_raw_xyz = []


def run():
    base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211025_2_group0013_351frames_71labeled'
    base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/tmp2'
    base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
    base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled/20211025_1_group0012_frame0050_labeled'
    base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'

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
        TI_radar_points = read_pcd(TI_pcd_path)     # (N, 5)
        lidar_json = load_json(calib_lidar)
        TI_json = load_json(TI_json_path)
        cam_json = load_json(img_json)

        # lidar2TI_anno = get_lidar2TI_anno(lidar_json, TI_json)
        print('TI_pcd_path:', TI_pcd_path)
        print('img_path:', img_path)
        print('TI_points: ', TI_radar_points.shape, '\n')
        Lidar_annotation_points = pts2rbev(lidar_json, TI_json, cam_json)



        if plot:
            plt.xlim(-20, 20)
            plt.ylim(0, 40)
            mask_pd = np.ones(TI_radar_points.shape[0], dtype=bool)
            mask_pd = np.logical_and(mask_pd, TI_radar_points[:, 1] <= 40)
            mask_pd = np.logical_and(mask_pd, TI_radar_points[:, 0] <= 20)
            mask_pd = np.logical_and(mask_pd, TI_radar_points[:, 0] >= -10)
            TI_points = TI_radar_points[mask_pd, :]
            TI_points = np.array(sorted(TI_points, key=lambda x: x[1], reverse=False))
            # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            # cmap_name = 'my_cmap'
            # cmp = LinearSegmentedColormap.from_list(cmap_name, colors, N=40)

            cmp = ListedColormap(['b', 'g', 'r'])
            norm = BoundaryNorm(TI_points[:, 1], len(TI_points[:, 1]))

            plt.scatter(TI_points[:, 0], TI_points[:, 1], s=5, marker='o', cmap=cmp, norm=norm)
            for i in range(len(Lidar_annotation_points)):
                x = Lidar_annotation_points[i][0]
                y = Lidar_annotation_points[i][1]
                plt.scatter(x, y, s=20, marker='o', c='k')
            plt.title(folder)
            save_name = '/media/personal_data/zhangq/RadarRGBFusionNet/results/' + folder + '.png'
            # plt.savefig(save_name, dpi=500)
            plt.show()
            plt.pause(0.1)
            plt.clf()




if __name__ == '__main__':
    run()

