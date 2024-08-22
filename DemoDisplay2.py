import os
import sys
# sys.path.insert(0, './yolov5')
sys.path.append('./Detection/yolov5')

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from Track.linear_assignment import my_assignment
from dataset.self_dataset2 import self_dataset2
from dataset.selfDatasetDisplay import self_dataset3
import openpyxl
import cv2
import json
from Detection.yolov5.utils.plots import Annotator
INFTY_COST = 1e+5

# sqrt((a-b)**2)
def pdist(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    # r2 = np.clip(r2, 0., float(np.inf))
    # make min value is 0, max value is inf
    r2 = np.sqrt(r2)
    return r2

def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data
def min_cost_matching(cost_matrix, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix[cost_matrix > max_distance] = INFTY_COST
    indices = linear_sum_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

def min_cost_matching2(cost_matrix, gt_tracks, estimated_tracks, track_indices=None, detection_indices=None):
    distance_threshold1 = 3.5
    if min(cost_matrix.shape) > 0:
        a = (cost_matrix <= distance_threshold1).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = my_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))


    unmatched_tracks = []
    for t, trk in enumerate(gt_tracks):
        if (t not in matched_indices[:, 0]):
            unmatched_tracks.append(t)

    unmatched_detections = []
    for d, det in enumerate(estimated_tracks):
        if (d not in matched_indices[:, 1]):
            unmatched_detections.append(d)


    matches = []

    # filter out matched with far distance
    distance_threshold2 = 3.5
    for m in matched_indices:
        if(cost_matrix[m[0], m[1]] > distance_threshold2):
            unmatched_tracks.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches)==0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return np.array(matches), unmatched_tracks, unmatched_detections

def undistort_image(image, rgb_json):
    intrinsic = rgb_json['intrinsic']
    radial_distortion = rgb_json['radial_distortion']
    tangential_distortion = rgb_json['tangential_distortion']
    k1, k2, k3 = radial_distortion
    p1, p2 = tangential_distortion

    image_undistort = cv2.undistort(image, np.array(intrinsic), np.array([k1, k2, p1, p2, k3]))
    # all_img = np.hstack((image, image_undistort))

    return image_undistort

def drawBox(img, bbox, color=[0, 0, 255]):
    lw = 1.5
    txt_color = (255, 255, 255)
    for i in range(len(bbox)):
        left_top = np.array([bbox[i][0], bbox[i][1]]).astype(int)
        right_bottom = np.array([bbox[i][0] + bbox[i][2], bbox[i][1] + bbox[i][3]]).astype(int)

        cv2.rectangle(img, left_top, right_bottom, color=color, thickness=5)
        depth = bbox[i][4]
        label = '{:.2f}'.format(depth) + 'm'
        # label = str(depth)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            # fontFace: ziti, fontScale:suofangxishu
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = left_top[1] - h - 3 >= 0  # label fits outside box
            right_bottom = left_top[0] + w, left_top[1] - h - 3 if outside else left_top[1] + h + 3
            cv2.rectangle(img, left_top, right_bottom, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (left_top[0], left_top[1] - 2 if outside else left_top[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

def tlwh2xyxy(bbox_tlwh):
    """
    TODO:
        Convert bbox from xtl_ytl_w_h to xc_yc_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    width, height = 511, 510
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x + w), width - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y + h), height - 1)
    return x1, y1, x2, y2


def run():

    max_distance = 5.0
    plot3 = True

    # Dataloader
    file_path = '/mnt/ChillDisk/personal_data/zhangq/RadarRGBFusionNet2_20231128/GroupPath/Display.xlsx'
    DataPath = openpyxl.load_workbook(file_path)
    ws = DataPath.active
    groups_excel = ws['A']
    datasets_path = []
    for cell in groups_excel:
        datasets_path.append(cell.value)
    data_base_path = '/mnt/ourDataset_v2/ourDataset_v2'

    # estimated_tracks=estimated_tracks, gt_tracks=gt_tracks, radar_detections=radar_detections,
    # img_detections=img_detections, detection2Dboxes=detection2Dboxes)
    # groupFile = np.load('./TrackResults/' + '20221217_group0016_mode3_602frames.npz')
    all_estimated_tracks = np.load(
        './TrackResults/all_estimated_tracks.npy')  # tracking result, (id,x,y,o, frame_num, group_num) 6
    all_gt_tracks = np.load('./TrackResults/all_gt_tracks.npy')  # gts, (id,x,y,z, classes, frame_num, group_num) 7
    all_radar_detections = np.load(
        './TrackResults/all_radar_detections.npy')  # radar dets, (x,y, frame_num, group_num) 4
    all_img_detections = np.load(
        './TrackResults/all_img_detections.npy')  # img proj dets, (x,y, frame_num, group_num) 4
    all_detection2Dboxes = np.load(
        './TrackResults/all_detection2Dboxes.npy')  # img 2D bbox (tlX, tlY, w, h, depth, frame_num, group_num)
    group_names = np.load('./TrackResults/group_names.npy')

    # for datasetNum in range(len(datasets_path)):
    #     data_base_path = '/mnt/ourDataset_v2/ourDataset_v2'
    #     source = os.path.join(data_base_path, datasets_path[datasetNum])
    #     dataset = self_dataset3(source)

    len_groups = int(max(all_gt_tracks[:, 6]))
    for num_group in range(len_groups+1):

        print('num_group', num_group)
        print('group_name', group_names[num_group])

        source = os.path.join(data_base_path, datasets_path[num_group])
        dataset = self_dataset3(source)

        group_name = group_names[num_group]
        # if num_group == 103:
        #     # print('num_group', num_group)
        #     print('*************'*10)
            # continue
        mask0 = np.ones(all_estimated_tracks.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_estimated_tracks[:, 5] == num_group)
        estimated_tracks = all_estimated_tracks[mask0, :]

        mask0 = np.ones(all_gt_tracks.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_gt_tracks[:, 6] == num_group)
        gt_tracks = all_gt_tracks[mask0, :]

        mask0 = np.ones(all_radar_detections.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_radar_detections[:, 3] == num_group)
        radar_detections = all_radar_detections[mask0, :]

        mask0 = np.ones(all_img_detections.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_img_detections[:, 3] == num_group)
        img_detections = all_img_detections[mask0, :]

        mask0 = np.ones(all_detection2Dboxes.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_detection2Dboxes[:, 6] == num_group)
        img_2DBbox = all_detection2Dboxes[mask0, :]

        if estimated_tracks.shape[0] == 0 or gt_tracks.shape[0] == 0 or radar_detections.shape[0] == 0 or img_detections.shape[0] == 0:
            continue

        num_min = int(min(gt_tracks[:, 5]))
        num_max = int(max(gt_tracks[:, 5]))
        print('length_lidar_frame:', num_max - num_min)


        if plot3:
            # title_name = group_name[0] + '_' + group_name[1]

            # figure1 = plt.figure()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            plt.ion()  # 必须打开交互模式
            fig.suptitle(group_name[1])
            cmap = plt.get_cmap('tab20b')
            maxLength = int(max(max(img_2DBbox[:, 5]), max(estimated_tracks[:, 4]), max(gt_tracks[:, 5]), max(radar_detections[:, 2]))) + 1
            for i in range(maxLength):
                mask_img = np.ones(img_2DBbox.shape[0], dtype=bool)
                mask_img = np.logical_and(mask_img, img_2DBbox[:, 5] == i)
                img_2DBboxCurr = img_2DBbox[mask_img, :]

                mask_img = np.ones(radar_detections.shape[0], dtype=bool)
                mask_img = np.logical_and(mask_img, radar_detections[:, 2] <= i)
                mask_img = np.logical_and(mask_img, radar_detections[:, 2] >= i-30)
                radar_detectionsCurr = radar_detections[mask_img, :]

                mask_img = np.ones(estimated_tracks.shape[0], dtype=bool)
                mask_img = np.logical_and(mask_img, estimated_tracks[:, 4] <= i)
                mask_img = np.logical_and(mask_img, estimated_tracks[:, 4] >= i-30)
                estimated_tracksCurr = estimated_tracks[mask_img, :]
                idsEstimate = set(estimated_tracksCurr[:, 0])  # curr and before estimate id

                mask_img = np.ones(gt_tracks.shape[0], dtype=bool)
                mask_img = np.logical_and(mask_img, gt_tracks[:, 5] <= i)
                mask_img = np.logical_and(mask_img, gt_tracks[:, 5] >= i-30)
                gt_tracksCurr = gt_tracks[mask_img, :]
                idsGT = set(gt_tracksCurr[:, 0])    # curr and before gt id

                CurrImgPath = dataset.rgb_data[i]
                # rgb_json_path = dataset.rgb_jsons[i]
                # rgb_json = load_json(rgb_json_path)
                CurrImg = cv2.imread(CurrImgPath)
                # print('img_2DBboxCurr', img_2DBboxCurr, '\n')
                drawBox(CurrImg, img_2DBboxCurr)
                # CurrImg = undistort_image(CurrImg, rgb_json)
                CurrImg = CurrImg[:, 225:736, :]
                # im0 = CurrImg.copy()
                # annotator = Annotator(im0, line_width=2, pil=not ascii)
                # label = ''
                # # annotator.box_label(bboxes, label, color=colors(c, True))
                # for currID in range(len(img_2DBboxCurr)):
                #     bboxes = tlwh2xyxy(img_2DBboxCurr[i][:4])
                #     annotator.box_label(bboxes, label, color=(255, 0, 0))

                ax1.cla()
                CurrImgRgb = cv2.cvtColor(CurrImg, cv2.COLOR_BGR2RGB)
                ax1.imshow(CurrImgRgb)
                ax1.axis('off')
                # cv2.imshow('CurrImg', CurrImg)
                # cv2.waitKey(10)
                # if cv2.waitKey(0) & 0xFF == 27:
                #     break

                ax2.cla()  # 清除旧图
                ax2.set_xlim(-20, 20)
                ax2.set_ylim(0, 100)

                for currEsID in idsEstimate:
                    mask_pd = np.ones(estimated_tracksCurr.shape[0], dtype=bool)
                    mask_pd = np.logical_and(mask_pd, estimated_tracksCurr[:, 0] == currEsID)
                    estimated_tracks_mask = estimated_tracksCurr[mask_pd, :]
                    # color = cmap(int(currEsID) % 20)
                    ax2.plot(estimated_tracks_mask[:, 1], estimated_tracks_mask[:, 2], color='r', alpha=1)

                for currGTID in idsGT:
                    mask_gd = np.ones(gt_tracksCurr.shape[0], dtype=bool)
                    mask_gd = np.logical_and(mask_gd, gt_tracksCurr[:, 0] == currGTID)
                    gt_tracks_mask = gt_tracksCurr[mask_gd, :]
                    # color = cmap(int(currGTID) % 20)
                    ax2.plot(gt_tracks_mask[:, 1], gt_tracks_mask[:, 2], color='k', linestyle='--', alpha=0.8)
                # plt.scatter(estimated_tracksCurr[:, 1], estimated_tracksCurr[:, 2], c='r', marker='.', alpha=1)  # 绘制新图
                # plt.scatter(gt_tracksCurr[:, 1], gt_tracksCurr[:, 2], c='k', marker='.', alpha=0.8)
                ax2.scatter(radar_detectionsCurr[:, 0], radar_detectionsCurr[:, 1], color='b', marker='s', s=5,
                            alpha=0.5)
                # ax2.legend()
                plt.pause(0.01)  # 使用pause 而不是show来显示 0.01是一个延迟时间
            plt.ioff()
            plt.close()
        print('***********************'*5, '\n')

    print('\ndone')

if __name__ == '__main__':
    run()