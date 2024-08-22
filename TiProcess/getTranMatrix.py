import os
# print('os.getcwd', os.getcwd)
import numpy as np
import pandas as pd
from TiProcess.BEV_TI import read_pcd, load_json, pts2rbev

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm


def IMU_Trans_TI(pointcloud, transform_matrix):
    '''
            transform pointcloud from coordinate1 to coordinate2 according to transform_matrix
        :param pointcloud: (x, y, z, ...)  (N, 3)
        :param transform_matrix: (4, 4) [R,T; 0,1]
        :return pointcloud_transformed: (x, y, z, ...)  (N, 3)
        '''
    n_points = pointcloud.shape[0]
    xyz = pointcloud[:, :3]
    xyz1 = np.vstack((xyz.T, np.ones((1, n_points))))   # (4, N)
    xyz1_transformed = np.matmul(transform_matrix, xyz1)
    pointcloud_transformed = np.hstack((
        xyz1_transformed[:3, :].T,
        pointcloud[:, 3:]
    ))
    return pointcloud_transformed


def pos2ecef(lon, lat, h):
    # WGS84长半轴
    a = 6378137
    # WGS84椭球扁率
    f = 1 / 298.257223563
    # WGS84短半轴
    b = a * (1 - f)
    # WGS84椭球第一偏心率
    e = np.sqrt(a * a - b * b) / a
    # WGS84椭球面卯酉圈的曲率半径
    N = a / np.sqrt(1 - e * e * np.sin(lat * np.pi / 180) * np.sin(lat * np.pi / 180))
    x_ecef = (N + h) * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y_ecef = (N + h) * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
    z_ecef = (N * (1 - (e * e)) + h) * np.sin(lat * np.pi / 180)

    return x_ecef, y_ecef, z_ecef

def pos2enu(lon, lat, h, lon_ref=0, lat_ref=0, h_ref=0):
    x_ecef, y_ecef, z_ecef = pos2ecef(lon, lat, h)
    x_ecef_ref, y_ecef_ref, z_ecef_ref = pos2ecef(lon_ref, lat_ref, h_ref)

    offset_x, offset_y, offset_z = x_ecef - x_ecef_ref, y_ecef - y_ecef_ref, z_ecef - z_ecef_ref

    sinLon = np.sin(lon_ref * np.pi / 180)
    cosLon = np.cos(lon_ref * np.pi / 180)
    sinLat = np.sin(lat_ref * np.pi / 180)
    cosLat = np.cos(lat_ref * np.pi / 180)

    x_enu = -1 * sinLon * offset_x + cosLon * offset_y
    y_enu = -1 * sinLat * cosLon * offset_x - 1 * sinLat * sinLon * offset_y + cosLat * offset_z
    z_enu = cosLat * cosLon * offset_x + cosLat * sinLon * offset_y + sinLat * offset_z

    xyz_enu = np.array([x_enu, y_enu, z_enu]).reshape(3, 1)

    return xyz_enu

def get_relative_time(TI_time):
    TI_time = float(TI_time)
    time_step = (data_time[1] - data_time[0]) / 2
    relative_time_id = -1
    for i in range(n):
        if abs(TI_time - data_time[i]) <= time_step:
            relative_time_id = i
            break

    return relative_time_id




def getTransMatrix(pre_time, curr_time):

    start_time = float(pre_time)
    end_time = float(curr_time)
    # Trans_matrix = np.array([])
    time_list = []
    for i in range(n):
        if data_time[i] >= start_time and data_time[i] < end_time:
            time_list.append(i)
        if data_time[i] >= end_time:
            break
    time_len = len(time_list)
    x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0
    if time_len != 0:
    #     for i in range(time_len - 1):
    #         j = i + 1
    #         dt = data_time[time_list[j]] - data_time[time_list[i]]
    #         ve += (data_ve[time_list[j]] + data_ve[time_list[i]]) / 2 * dt
    #         vn += (data_vn[time_list[j]] + data_vn[time_list[i]]) / 2 * dt
    #         vu += (data_vu[time_list[j]] + data_vu[time_list[i]]) / 2 * dt

            # roll += (data_roll[j] + data_roll[i]) / 2 * dt
            # pitch += (data_pitch[j] + data_pitch[i]) / 2 * dt
            # yaw += (data_yaw[j] + data_yaw[i]) / 2 * dt

        dt = data_time[time_list[-1]] - data_time[time_list[0]]
        x = (data_ve[time_list[-1]] + data_ve[time_list[0]]) / 2 * dt
        y = (data_vn[time_list[-1]] + data_vn[time_list[0]]) / 2 * dt
        z = (data_vu[time_list[-1]] + data_vu[time_list[0]]) / 2 * dt
        roll = data_roll[time_list[-1]] - data_roll[time_list[0]]
        pitch = data_pitch[time_list[-1]] - data_pitch[time_list[0]]
        yaw = data_yaw[time_list[-1]] - data_yaw[time_list[0]]

        R1 = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        R2 = [[1, 0, 0], [0, np.cos(pitch), np.sin(pitch)], [0, -np.sin(pitch), np.cos(pitch)]]
        R3 = [[np.cos(roll), 0, -np.sin(roll)], [0, 1, 0], [np.sin(roll), 0, np.cos(roll)]]

        R = np.dot(np.dot(R1, R2), R3)
        T = np.array([x, y, z]).reshape(3, 1)
        # Trans_matrix = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1])))

    return R, T

def getTransMatrix2(pre_time_id, curr_time_id):

    dt = data_time[curr_time_id] - data_time[pre_time_id]
    x = (data_ve[curr_time_id] + data_ve[pre_time_id]) / 2 * dt
    y = (data_vn[curr_time_id] + data_vn[pre_time_id]) / 2 * dt
    z = (data_vu[curr_time_id] + data_vu[pre_time_id]) / 2 * dt
    roll = data_roll[curr_time_id] - data_roll[pre_time_id]
    pitch = data_pitch[curr_time_id] - data_pitch[pre_time_id]
    yaw = data_yaw[curr_time_id] - data_yaw[pre_time_id]

    R1 = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    R2 = [[1, 0, 0], [0, np.cos(pitch), np.sin(pitch)], [0, -np.sin(pitch), np.cos(pitch)]]
    R3 = [[np.cos(roll), 0, -np.sin(roll)], [0, 1, 0], [np.sin(roll), 0, np.cos(roll)]]

    R = np.dot(np.dot(R1, R2), R3)
    T = np.array([x, y, z]).reshape(3, 1)
    # Trans_matrix = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1])))

    return R, T

data = pd.read_csv('./runs/heading_imu_input.csv')
n = data.shape[0]
data_time = data['unix_time'].values
curr_time = data_time[0]


data_lon = data['longitude'].values
data_lat = data['latitude'].values
data_h = data['height'].values
data_ve = data['ve(m/s)'].values
data_vn = data['vn(m/s)'].values
data_vu = data['vu(m/s)'].values
data_roll = data['roll(rad)'].values
data_pitch = data['pitch(rad)'].values
data_yaw = data['yaw(rad)'].values
# start_time = 1635144526.0
# end_time = 1635144526.5

def run():
    base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'

    folders = os.listdir(base_path)
    folders = sorted(folders)
    frame_num = -1
    plot = True
    pre_TI_timestamp = 0
    pre_TI_points = []
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
        print('pre_TI_timestamp:', pre_TI_timestamp)
        print('TI_timestamp:', TI_timestamp)
        print('TI_points: ', TI_points.shape, '\n')

        if frame_num != 0:

            # R, T = getTransMatrix(pre_TI_timestamp, TI_timestamp)

            pre_relative_time_id = get_relative_time(pre_TI_timestamp)
            curr_relative_time_id = get_relative_time(TI_timestamp)
            R, T = getTransMatrix2(pre_relative_time_id, curr_relative_time_id)

            pre_enu_xyz = pos2enu(data_lon[pre_relative_time_id], data_lat[pre_relative_time_id], data_h[pre_relative_time_id])
            curr_enu_xyz = pos2enu(data_lon[curr_relative_time_id], data_lat[curr_relative_time_id], data_h[curr_relative_time_id])
            pre_enu_TI_points = pre_TI_points.T + pre_enu_xyz

            # new_TI_points = (pre_enu_TI_points - T - curr_enu_xyz).T
            new_TI_points = (np.dot(np.linalg.inv(R), (pre_enu_TI_points - T)) - curr_enu_xyz).T
            # pre_TI_points = pre_TI_points.T
            # new_TI_points = IMU_Trans_TI(pre_TI_points, Trans_matrix)

            if plot:
                plt.xlim(-20, 20)
                plt.ylim(0, 40)
                mask_pd = np.ones(TI_points.shape[0], dtype=bool)
                mask_pd = np.logical_and(mask_pd, TI_points[:, 1] <= 40)
                mask_pd = np.logical_and(mask_pd, TI_points[:, 0] <= 20)
                mask_pd = np.logical_and(mask_pd, TI_points[:, 0] >= -10)
                TI_points = TI_points[mask_pd, :]
                # TI_points = np.array(sorted(TI_points, key=lambda x: x[1], reverse=False))
                # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
                # cmap_name = 'my_cmap'
                # cmp = LinearSegmentedColormap.from_list(cmap_name, colors, N=40)

                # cmp = ListedColormap(['b', 'g', 'r'])
                # norm = BoundaryNorm(TI_points[:, 1], len(TI_points[:, 1]))

                plt.scatter(TI_points[:, 0], TI_points[:, 1], s=5, marker='o', c='b')

                mask_pd = np.ones(new_TI_points.shape[0], dtype=bool)
                mask_pd = np.logical_and(mask_pd, new_TI_points[:, 1] <= 40)
                mask_pd = np.logical_and(mask_pd, new_TI_points[:, 0] <= 20)
                mask_pd = np.logical_and(mask_pd, new_TI_points[:, 0] >= -10)
                new_TI_points = new_TI_points[mask_pd, :]
                # new_TI_points = np.array(sorted(new_TI_points, key=lambda x: x[1], reverse=False))
                plt.scatter(new_TI_points[:, 0], new_TI_points[:, 1], s=5, marker='*', c='g')

                mask_pd = np.ones(pre_TI_points.shape[0], dtype=bool)
                mask_pd = np.logical_and(mask_pd, pre_TI_points[:, 1] <= 40)
                mask_pd = np.logical_and(mask_pd, pre_TI_points[:, 0] <= 20)
                mask_pd = np.logical_and(mask_pd, pre_TI_points[:, 0] >= -10)
                pre_TI_points = pre_TI_points[mask_pd, :]
                # new_TI_points = np.array(sorted(new_TI_points, key=lambda x: x[1], reverse=False))
                plt.scatter(pre_TI_points[:, 0], pre_TI_points[:, 1], s=5, marker='x', c='r')

                for i in range(len(Lidar_annotation_points)):
                    x = Lidar_annotation_points[i][0]
                    y = Lidar_annotation_points[i][1]
                    plt.scatter(x, y, s=50, marker='o', c='k')
                plt.title(folder)
                plt.show()
                plt.pause(0.1)
                plt.clf()

        pre_TI_timestamp = TI_timestamp
        pre_TI_points = TI_points


if __name__ == '__main__':
    run()
    print('done')




