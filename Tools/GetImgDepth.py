
import os
import numpy as np
import cv2
import torch
import openpyxl



def Radar2img(points, OCU_json, cam_json):

    V = np.array(cam_json['intrinsic'])
    V0 = np.hstack((V, np.ones((3, 1))))        # 3*4
    RT = np.array(OCU_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])  # 4*4, OCULiiRadar_to_LeopardCamera0_extrinsic, TIRadar_to_LeopardCamera0_extrinsic
    VRT = np.matmul(V0, RT)
    # length = points.shape[0]
    # xyz = np.concatenate([points[:, 0].reshape((length, 1)),
    #                       points[:, 2].reshape((length, 1)),
    #                       points[:, 1].reshape((length, 1))], axis=1).T # 3*N
    xyz = points[:, 0:3].T     # 3*N
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])  # 给xyz在后面叠加了一行1 (4,N)
    uvd_img = np.matmul(VRT, xyz1)  # (3,N)
    uvd_img[0, :] = uvd_img[0, :] / uvd_img[2, :]
    uvd_img[1, :] = uvd_img[1, :] / uvd_img[2, :]

    return uvd_img



def getUVDistance(bbox_centers, radar_uvds):
    """
        bbox_centers: (N, 2): bboxes center's uv
        radar_uvds: (3, M): radar points's uvd
     """
    radar_uvs = radar_uvds[0:2, :].T  # (M, 2)
    a, b = np.asarray(bbox_centers), np.asarray(radar_uvs)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    # r2 = np.clip(r2, 0., float(np.inf))
    # make min value is 0, max value is inf
    r2 = np.sqrt(r2)
    return r2

def getImgDepth(bbox_xywhs, rgb_json, OCU_json, OCU_data):
    min_num = 1
    LeopardCamera0_IntrinsicMatrix = np.array(rgb_json['intrinsic'])  # 3*3
    OCU_to_LeopardCamera0_TransformMatrix = np.array(OCU_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])  # 4*4
    R = OCU_to_LeopardCamera0_TransformMatrix[0:3, 0:3]  # 3*3
    T = OCU_to_LeopardCamera0_TransformMatrix[0:3, 3].reshape(3, 1)  # 3*1
    radar_uvd = Radar2img(OCU_data, OCU_json, rgb_json)  # (3, M)
    radar_num = radar_uvd.shape[1]

    bbox_center = bbox_xywhs[:, 0:2]  # (N, 2)
    bbox_num = bbox_xywhs.shape[0]
    bbox_radar_distances = getUVDistance(bbox_center, radar_uvd)  # (N, M)
    bbox_center_uvd = np.concatenate((bbox_center, np.zeros((bbox_num, 1))), axis=1)  # (N, 3)

    for i in range(bbox_num):
        bbox_radar_distance = bbox_radar_distances[i]
        mins_uv_indexes = np.argpartition(bbox_radar_distance, min_num)  # min_num indexes
        mins_d = radar_uvd[2, mins_uv_indexes[:min_num]]
        mean_d = np.mean(mins_d)
        bbox_center_uvd[i, 2] = mean_d

    bbox_center_uvd[:, 0] = bbox_center_uvd[:, 0] * bbox_center_uvd[:, 2]
    bbox_center_uvd[:, 1] = bbox_center_uvd[:, 1] * bbox_center_uvd[:, 2]
    bbox_center_uvd = bbox_center_uvd.T  # (3, N)


    XYZ_camera = np.matmul(np.linalg.inv(LeopardCamera0_IntrinsicMatrix), bbox_center_uvd)  # 3*N
    XYZ_OCU = np.matmul(np.linalg.inv(R), XYZ_camera - T)  # 3*N
    XYZ_OCU = XYZ_OCU.T
    XYZ_OCU = np.concatenate([XYZ_OCU[:, 0].reshape((bbox_num, 1)),
                              XYZ_OCU[:, 2].reshape((bbox_num, 1)),
                              XYZ_OCU[:, 1].reshape((bbox_num, 1))], axis=1)  # 3*N

    XYZ_OCU = XYZ_OCU[:, 0:2]
    return XYZ_OCU



if __name__ == '__main__':
    print('done')