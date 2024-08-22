import os

import numpy as np
from torch.utils.data import Dataset
import cv2
import json
import matplotlib.pyplot as plt
import re
import warnings
# import mayavi.mlab


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
    # print(f'pcd points: {points.shape}')

    # if pts_view:
    #     ptsview(points)
    return points


# def ptsview(points):
#     x, y, z = points[:, 0], points[:, 1], points[:, 2]
#     d = np.sqrt(x ** 2 + y ** 2)
#     vals = 'height'
#     if vals == 'height':
#         col = z
#     else:
#         col = d
#     # f = mayavi.mlab.gcf()
#     # camera = f.scene.camera
#     # camera.yaw(90)
#     fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
#     # camera = fig.scene.camera
#     # camera.yaw(90)
#     # cam, foc = mayavi.mlab.move()
#     # print(cam, foc)
#     mayavi.mlab.points3d(x, y, z,
#                          col,
#                          mode='point',
#                          colormap='spectral',
#                          figure=fig)
#     mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
#     axes = np.array(
#         [[20, 0, 0, ], [0, 20, 0], [0, 0, 20]]
#     )
#     mayavi.mlab.plot3d(
#         [0, axes[0, 0]],
#         [0, axes[0, 1]],
#         [0, axes[0, 2]],
#         color=(1, 0, 0),
#         tube_radius=None,
#         figure=fig
#     )
#     mayavi.mlab.plot3d(
#         [0, axes[1, 0]],
#         [0, axes[1, 1]],
#         [0, axes[1, 2]],
#         color=(0, 1, 0),
#         tube_radius=None,
#         figure=fig
#     )
#     mayavi.mlab.plot3d(
#         [0, axes[2, 0]],
#         [0, axes[2, 1]],
#         [0, axes[2, 2]],
#         color=(0, 0, 1),
#         tube_radius=None,
#         figure=fig
#     )
#     mayavi.mlab.show()
