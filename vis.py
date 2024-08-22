import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import cv2
import matplotlib.patches as patches

def load_image(path):
    image_bgr = cv2.imread(path)
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def undistort_image(image, image_info):
    intrinsic = image_info['intrinsic']
    radial_distortion = image_info['radial_distortion']
    tangential_distortion = image_info['tangential_distortion']
    k1, k2, k3 = radial_distortion
    p1, p2 = tangential_distortion

    image_undistorted = cv2.undistort(image, np.array(intrinsic), np.array([k1, k2, p1, p2, k3]))

    return image_undistorted

def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data

def load_OCULiiRadar_pcd(path: str) -> dict:
    with open(path, "r") as f:
        data = f.readlines()
    data = data[10:]
    data = np.array(data)
    data = np.char.replace(data, '\n', '')
    data = np.char.split(data, ' ')
    data = np.array(list(data))

    x = np.array(data[:, 0], dtype=np.float32)
    y = np.array(data[:, 1], dtype=np.float32)
    z = np.array(data[:, 2], dtype=np.float32)
    doppler = np.array(data[:, 3], dtype=np.float32)
    snr = np.array(data[:, 4], dtype=np.float32)

    pcd = {
        'x': x,
        'y': y,
        'z': z,
        'doppler': doppler,
        'snr': snr
    }
    return pcd

def projection(x, y, z, intrinsic, extrinsic):
    n = x.shape[0]

    xyz1 = np.stack((x, y, z, np.ones((n, ))))

    xyz1_ = np.matmul(extrinsic, xyz1)
    mask_front = xyz1_[2, :] > 0
    xyz_front = xyz1_[:3, mask_front]

    UVZ = np.matmul(intrinsic, xyz_front)
    uv1 = UVZ[0:2, :] / UVZ[2, :]

    u = uv1[0, :].round().astype('int')
    v = uv1[1, :].round().astype('int')
    depth = UVZ[2, :]

    return u, v, depth, mask_front

def pcd_in_image(pcd: dict, image_height: int, image_width: int, intrinsic:np.array, extrinsic: np.array) -> dict:
    assert 'x' in pcd.keys() and 'y' in pcd.keys() and 'z' in pcd.keys()

    u, v, depth, mask_front = projection(pcd['x'], pcd['y'], pcd['z'], intrinsic, extrinsic)

    mask_u = np.logical_and(u >= 0, u < image_width)

    mask_v = np.logical_and(v >= 0, v < image_height)

    mask = np.logical_and(mask_u, mask_v)

    res = dict()
    for key, value in pcd.items():
        res[key] = value[mask_front][mask]
    res['u'] = u[mask]
    res['v'] = v[mask]
    res['depth'] = depth[mask]

    return res

class TrackingResult(object):
    def __init__(self, **kwargs):

        self.root_dataset = kwargs['root_dataset']

        self.group_names = pd.DataFrame(np.load(kwargs['path_group_names']), columns=['group_id', 'group_name'])
        self.group_names['group_id'] = self.group_names['group_id'].astype(int)
        self.group_names['group_name'] = self.group_names['group_name'].astype(str)

        self.num_groups = self.group_names.shape[0]
        self.num_frames_in_group = [int(item.split('_')[-1].replace('frames', ''))
                                    for item in self.group_names['group_name'].values]

        self.tracks_estimated = pd.DataFrame(np.load(kwargs['path_tracks_estimated']), columns=['track_id', 'x', 'y', 'ori', 'frame_id', 'group_id'])
        self.tracks_estimated['track_id'] = self.tracks_estimated['track_id'].astype(int)
        self.tracks_estimated['x'] = self.tracks_estimated['x'].astype(float)
        self.tracks_estimated['y'] = self.tracks_estimated['y'].astype(float)
        self.tracks_estimated['ori'] = self.tracks_estimated['ori'].astype(float)
        self.tracks_estimated['frame_id'] = self.tracks_estimated['frame_id'].astype(int)
        self.tracks_estimated['group_id'] = self.tracks_estimated['group_id'].astype(int)

        self.tracks_gt = pd.DataFrame(np.load(kwargs['path_tracks_gt']), columns=['track_id', 'x', 'y', 'z', 'class_id', 'frame_id', 'group_id'])
        self.tracks_gt['track_id'] = self.tracks_gt['track_id'].astype(int)
        self.tracks_gt['x'] = self.tracks_gt['x'].astype(float)
        self.tracks_gt['y'] = self.tracks_gt['y'].astype(float)
        self.tracks_gt['z'] = self.tracks_gt['z'].astype(float)
        self.tracks_gt['class_id'] = self.tracks_gt['class_id'].astype(int)
        self.tracks_gt['frame_id'] = self.tracks_gt['frame_id'].astype(int)
        self.tracks_gt['group_id'] = self.tracks_gt['group_id'].astype(int)
        # insert tracks_gt
        for group_id in range(self.num_groups):
            num_frames = self.num_frames_in_group[group_id]
            track_ids = np.unique(self.tracks_gt['track_id'].values)
            track_start = self.tracks_gt.groupby(['track_id'])['frame_id'].min().values
            track_end = self.tracks_gt.groupby(['track_id'])['frame_id'].max().values
            for frame_id in range(num_frames):
                for i, track_id in enumerate(track_ids):
                    if frame_id >= track_start[i] and frame_id <= track_end[i]:
                        if self.tracks_gt[
                            (self.tracks_gt['track_id'] == track_id) &
                            (self.tracks_gt['frame_id'] == frame_id) &
                            (self.tracks_gt['group_id'] == group_id)
                        ].shape[0] != 1:
                            diff = self.tracks_gt[
                                       (self.tracks_gt['track_id'] == track_id) & (self.tracks_gt['group_id'] == group_id)
                                   ]['frame_id'] - frame_id
                            frame_id_pre = diff[diff < 0].abs().idxmin()
                            frame_id_next = diff[diff > 0].abs().idxmin()
                            new_x = np.interp(
                                frame_id,
                                np.array([self.tracks_gt.loc[frame_id_pre]['frame_id'], self.tracks_gt.loc[frame_id_next]['frame_id']]),
                                np.array([self.tracks_gt.loc[frame_id_pre]['x'], self.tracks_gt.loc[frame_id_next]['x']])
                            )
                            new_y = np.interp(
                                frame_id,
                                np.array([self.tracks_gt.loc[frame_id_pre]['frame_id'], self.tracks_gt.loc[frame_id_next]['frame_id']]),
                                np.array([self.tracks_gt.loc[frame_id_pre]['y'], self.tracks_gt.loc[frame_id_next]['y']])
                            )
                            new_z = np.interp(
                                frame_id,
                                np.array([self.tracks_gt.loc[frame_id_pre]['frame_id'], self.tracks_gt.loc[frame_id_next]['frame_id']]),
                                np.array([self.tracks_gt.loc[frame_id_pre]['z'], self.tracks_gt.loc[frame_id_next]['z']])
                            )
                            new_row = pd.DataFrame({
                                'track_id': [track_id],
                                'x': [new_x],
                                'y': [new_y],
                                'z': [new_z],
                                'class_id': [self.tracks_gt.loc[frame_id_pre]['class_id']],
                                'frame_id': [frame_id],
                                'group_id': [group_id]
                            })
                            index_position = ((self.tracks_gt['group_id'] == group_id) & (self.tracks_gt['frame_id'] > frame_id)).values.argmax()
                            self.tracks_gt = pd.concat(
                                [self.tracks_gt.iloc[:index_position], new_row, self.tracks_gt.iloc[index_position:]]
                            ).reset_index(drop=True)
        
        self.detections_radar = pd.DataFrame(np.load(kwargs['path_detections_radar']), columns=['x', 'y', 'frame_id', 'group_id'])
        self.detections_radar['x'] = self.detections_radar['x'].astype(float)
        self.detections_radar['y'] = self.detections_radar['y'].astype(float)
        self.detections_radar['frame_id'] = self.detections_radar['frame_id'].astype(int)
        self.detections_radar['group_id'] = self.detections_radar['group_id'].astype(int)

        self.detections_camera = pd.DataFrame(np.load(kwargs['path_detections_camera']), columns=['x', 'y', 'frame_id', 'group_id'])
        self.detections_camera['x'] = self.detections_camera['x'].astype(float)
        self.detections_camera['y'] = self.detections_camera['y'].astype(float)
        self.detections_camera['frame_id'] = self.detections_camera['frame_id'].astype(int)
        self.detections_camera['group_id'] = self.detections_camera['group_id'].astype(int)

        self.detections_2Dbbox = pd.DataFrame(np.load(kwargs['path_detections_2Dbbox']), columns=['u', 'v', 'w', 'h', 'depth', 'frame_id', 'group_id'])
        self.detections_2Dbbox['u'] = self.detections_2Dbbox['u'].astype(int)
        self.detections_2Dbbox['v'] = self.detections_2Dbbox['v'].astype(int)
        self.detections_2Dbbox['w'] = self.detections_2Dbbox['w'].astype(int)
        self.detections_2Dbbox['h'] = self.detections_2Dbbox['h'].astype(int)
        self.detections_2Dbbox['depth'] = self.detections_2Dbbox['depth'].astype(float)
        self.detections_2Dbbox['frame_id'] = self.detections_2Dbbox['frame_id'].astype(int)
        self.detections_2Dbbox['group_id'] = self.detections_2Dbbox['group_id'].astype(int)

    def get_data(self, group_id, frame_id, num_frames_history=30):
        group_name = self.group_names[self.group_names['group_id'] == group_id]['group_name'][0]
        frame_name = 'frame{:>04d}'.format(frame_id)
        root_data = os.path.join(self.root_dataset, group_name, frame_name)
        assert os.path.exists(root_data)

        camera = 'LeopardCamera0'
        image_raw = load_image(glob.glob(os.path.join(root_data, camera, '*.png'))[0])
        image_info = load_json(glob.glob(os.path.join(root_data, camera, '*.json'))[0])
        intrinsic = np.array(image_info['intrinsic'])
        image_undistorted = undistort_image(image_raw, image_info)
        # crop
        image = image_undistorted[:, 225:736, :]

        pcd_sensor = 'OCULiiRadar'
        pcd = load_OCULiiRadar_pcd(glob.glob(os.path.join(root_data, pcd_sensor, '*.pcd'))[0])
        pcd = pd.DataFrame(pcd)
        pcd_info = load_json(glob.glob(os.path.join(root_data, pcd_sensor, '*.json'))[0])
        extrinsic = np.array(pcd_info['{}_to_{}_extrinsic'.format(pcd_sensor, camera)])

        tracks_estimated = self.tracks_estimated[(self.tracks_estimated['group_id'] == group_id) & (self.tracks_estimated['frame_id'] == frame_id)]
        tracks_estimated_history = self.tracks_estimated[
            (self.tracks_estimated['group_id'] == group_id) &
            (self.tracks_estimated['frame_id'] <= frame_id) &
            (self.tracks_estimated['frame_id'] > frame_id - num_frames_history)
        ]
        tracks_estimated_history = tracks_estimated_history.sort_values(by=['frame_id', 'track_id'])

        tracks_gt = self.tracks_gt[(self.tracks_gt['group_id'] == group_id) & (self.tracks_gt['frame_id'] == frame_id)]
        tracks_gt_history = self.tracks_gt[
            (self.tracks_gt['group_id'] == group_id) &
            (self.tracks_gt['frame_id'] <= frame_id) &
            (self.tracks_gt['frame_id'] > frame_id - num_frames_history)
        ]
        tracks_gt_history = tracks_gt_history.sort_values(by=['frame_id', 'track_id'])

        detections_radar = self.detections_radar[(self.detections_radar['group_id'] == group_id) & (self.detections_radar['frame_id'] == frame_id)]
        detections_camera = self.detections_camera[(self.detections_camera['group_id'] == group_id) & (self.detections_camera['frame_id'] == frame_id)]
        detections_2Dbbox = self.detections_2Dbbox[(self.detections_2Dbbox['group_id'] == group_id) & (self.detections_2Dbbox['frame_id'] == frame_id)]
        # after cropped
        detections_2Dbbox['u'] = detections_2Dbbox['u'] - 225

        data = {
            'group_name': group_name,
            'frame_name': frame_name,
            'image': image,
            'intrinsic': intrinsic,
            'pcd': pcd,
            'extrinsic': extrinsic,
            'tracks_estimated': tracks_estimated,
            'tracks_estimated_history': tracks_estimated_history,
            'tracks_gt': tracks_gt,
            'tracks_gt_history': tracks_gt_history,
            'detections_radar': detections_radar,
            'detections_camera': detections_camera,
            'detections_2Dbbox': detections_2Dbbox
        }

        return data

def main():
    params = {
        'root_dataset': '/mnt/ourDataset_v2/ourDataset_v2',
        'path_group_names': './TrackResults/group_names.npy',
        'path_tracks_estimated': './TrackResults/all_estimated_tracks.npy',
        'path_tracks_gt': './TrackResults/all_gt_tracks.npy',
        'path_detections_radar': './TrackResults/all_radar_detections.npy',
        'path_detections_camera': './TrackResults/all_img_detections.npy',
        'path_detections_2Dbbox': './TrackResults/all_detection2Dboxes.npy'
    }
    root_output = './runs'
    dynamic = True

    if not os.path.exists(root_output):
        os.mkdir(root_output)
        print('create {}'.format(root_output))

    tracking_result = TrackingResult(**params)

    w_vedio = 1920
    h_vedio = 1080
    dpi_vedio = 100

    w_picture = 880
    h_picture = 880
    w_colorbar = 10
    padding_top = 100
    padding_bottom = 100
    padding_left = 10
    padding_mid = 50
    padding_colorbar = 10
    padding_right = 80
    w_all = padding_left + w_picture + padding_mid + w_picture + padding_colorbar + w_colorbar + padding_right
    h_all = padding_top + h_picture + padding_bottom

    fig = plt.figure(figsize=(w_vedio / dpi_vedio, h_vedio / dpi_vedio), facecolor='white')
    if dynamic:
        plt.ion()

    num_groups = tracking_result.num_groups
    for idx_group in range(num_groups):
        num_frames_in_group = tracking_result.num_frames_in_group[idx_group]

        output_path = os.path.join(root_output, '{}.mp4'.format(tracking_result.group_names['group_name'][idx_group]))
        writer = FFMpegWriter(fps=10)
        with writer.saving(fig, output_path, dpi=dpi_vedio):

            for idx_frame in range(num_frames_in_group):
                data = tracking_result.get_data(idx_group, idx_frame)
                print('=' * 100)
                print('({}/{}){}, ({}/{})/{}'.format(
                    idx_group + 1, num_groups, data['group_name'],
                    idx_frame + 1, num_frames_in_group, data['frame_name']
                ))

                # camera image
                left1 = padding_left / w_all
                bottom1 = padding_bottom / h_all
                w1 = w_picture / w_all
                h1 = h_picture / h_all
                ax1 = fig.add_axes([left1, bottom1, w1, h1])
                ax1.imshow(data['image'])
                ax1.axis('off')
                ax1.set_title('Camera {}/{}'.format(idx_frame + 1, num_frames_in_group))

                # 2Dbbox
                if data['detections_2Dbbox'].shape[0] > 0:
                    print('draw detections_2Dbbox')
                    print(data['detections_2Dbbox'])
                    for i in range(data['detections_2Dbbox'].shape[0]):
                        rect = patches.Rectangle(
                            (data['detections_2Dbbox']['u'].iloc[i],
                             data['detections_2Dbbox']['v'].iloc[i]),
                            data['detections_2Dbbox']['w'].iloc[i],
                            data['detections_2Dbbox']['h'].iloc[i],
                            linewidth=1, edgecolor='r', facecolor='none'
                        )
                        ax1.add_patch(rect)
                        ax1.text(
                            data['detections_2Dbbox']['u'].iloc[i],
                            data['detections_2Dbbox']['v'].iloc[i],
                            '{:.3f} m'.format(data['detections_2Dbbox']['depth'].iloc[i]),
                            color='r', fontsize=14, ha='left', va='bottom'
                        )

                # radar pcd
                left2 = left1 + w1 + padding_mid / w_all
                bottom2 = padding_bottom / h_all
                w2 = w_picture / w_all
                h2 = h_picture / h_all
                ax2 = fig.add_axes([left2, bottom2, w2, h2])
                cmap = 'jet'
                velocity_min, velocity_max = -10, 10
                ax2.scatter(
                    x=data['pcd']['x'].values,
                    y=data['pcd']['z'].values,
                    s=np.round(data['pcd']['snr'].values),
                    c=np.round(data['pcd']['doppler'].values),
                    cmap=cmap, vmin=velocity_min, vmax=velocity_max, alpha=0.5, label='_radar_pcd'
                )
                ax2.set_xlim(-20, 20)
                ax2.set_ylim(0, 100)
                ax2.set_xlabel('x/m')
                ax2.set_ylabel('y/m')
                ax2.grid(color='gray', linestyle='-', linewidth=1)
                ax2.set_aspect(1)
                ax2.set_title('Radar {}/{}'.format(idx_frame + 1, num_frames_in_group))

                # tracking results
                if data['tracks_estimated_history'].shape[0] > 0:
                    print('draw tracks_estimated_history')
                    print(data['tracks_estimated_history'])
                    track_ids = np.unique(data['tracks_estimated_history']['track_id'].values)
                    for i, track_id in enumerate(track_ids):
                        if track_id not in np.unique(data['tracks_estimated']['track_id'].values):
                            continue

                        track = data['tracks_estimated_history'][data['tracks_estimated_history']['track_id'] == track_id]
                        print('track_id={}, num={}'.format(track_id, track.shape[0]))
                        if i == 0:
                            ax2.plot(track['x'].values, track['y'].values, 'k.-', label='track_estimated')
                        else:
                            ax2.plot(track['x'].values, track['y'].values, 'k.-')

                if data['tracks_estimated'].shape[0] > 0:
                    print('draw tracks_estimated')
                    print(data['tracks_estimated'])
                    track_ids = np.unique(data['tracks_estimated']['track_id'].values)
                    for i, track_id in enumerate(track_ids):
                        track = data['tracks_estimated'][data['tracks_estimated']['track_id'] == track_id]
                        if i == 0:
                            ax2.scatter(track['x'].values, track['y'].values, s=100, color='r', marker='X', label='position_estimated')
                        else:
                            ax2.scatter(track['x'].values, track['y'].values, s=100, color='r', marker='X')

                if data['tracks_gt_history'].shape[0] > 0:
                    print('draw tracks_gt_history')
                    print(data['tracks_gt_history'])
                    track_ids = np.unique(data['tracks_gt_history']['track_id'].values)
                    for i, track_id in enumerate(track_ids):
                        if track_id not in np.unique(data['tracks_gt']['track_id'].values):
                            continue

                        track = data['tracks_gt_history'][data['tracks_gt_history']['track_id'] == track_id]
                        print('track_id={}, num={}'.format(track_id, track.shape[0]))
                        if i == 0:
                            ax2.plot(track['x'].values, track['y'].values, 'k.--', label='track_gt')
                        else:
                            ax2.plot(track['x'].values, track['y'].values, 'k.--')

                if data['tracks_gt'].shape[0] > 0:
                    print('draw tracks_gt')
                    print(data['tracks_gt'])
                    track_ids = np.unique(data['tracks_gt']['track_id'].values)
                    for i, track_id in enumerate(track_ids):
                        track = data['tracks_gt'][data['tracks_gt']['track_id'] == track_id]
                        if i == 0:
                            ax2.scatter(track['x'].values, track['y'].values, s=100, color='b', marker='X', label='position_gt')
                        else:
                            ax2.scatter(track['x'].values, track['y'].values, s=100, color='b', marker='X')

                plt.legend(loc=2)

                left3 = left2 + w2 + padding_colorbar / w_all
                bottom3 = padding_bottom / h_all
                w3 = w_colorbar / w_all
                h3 = h_picture / h_all
                plt.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=velocity_min, vmax=velocity_max)),
                    cax=fig.add_axes([left3, bottom3, w3, h3]),
                    orientation='vertical', label='velocity(m/s)'
                )

                writer.grab_frame()

                if dynamic:
                    plt.pause(0.1)
                    plt.clf()
                else:
                    plt.show()
                    fig = plt.figure(figsize=(w_vedio / dpi_vedio, h_vedio / dpi_vedio), facecolor='white')





    print('done')


if __name__ == '__main__':
    main()

