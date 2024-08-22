import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from Track.linear_assignment import my_assignment

import sys

import pandas as pd
import openpyxl

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

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

def pdistRMSE(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    # r2 = np.clip(r2, 0., float(np.inf))
    # make min value is 0, max value is inf
    return r2

def min_cost_matching(cost_matrix, tracks, detections, max_distance, track_indices=None, detection_indices=None):
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

def min_cost_matching2(cost_matrix, gt_tracks, estimated_tracks, max_distance, track_indices=None, detection_indices=None):
    distance_threshold1 = 2.0
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
    distance_threshold2 = 2.5
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

def getMOTPandRMSE(gt_detections, object_detections, all_meanP, all_RMSE, Pre, Rec):

    mask_gd = np.ones(gt_detections.shape[0], dtype=bool)
    mask_pd = np.ones(object_detections.shape[0], dtype=bool)
    cost, costRMSE = [], []
    groundtruth_number = 0
    TP, FP, FN = 0, 0, 0

    for i in range(int(max(object_detections[:, 2])) + 1):
        # 选择时刻
        mask1 = np.logical_and(mask_pd, object_detections[:, 2] == i)
        object_detections1 = object_detections[mask1, :]
        object_detections2 = object_detections1[:, [0, 1]]  # 预测值的x, y

        mask2 = np.logical_and(mask_gd, gt_detections[:, 5] == i)
        gt_detections1 = gt_detections[mask2, :]
        groundtruth_number += gt_detections1.shape[0]
        gt_detections2 = gt_detections1[:, [1, 2]]  # 真实值的x, y

        # 成本矩阵
        costmatrix = pdist(gt_detections2, object_detections2)
        costmatrixRMSE = pdistRMSE(gt_detections2, object_detections2)
        # 最小成本分配
        matches, unmatched_gd, unmatched_pd = min_cost_matching(costmatrix, gt_detections2,
                                                                object_detections2, max_distance=6.0)
        TP += len(matches)
        FP += len(unmatched_pd)
        FN += len(unmatched_gd)
        # matches, unmatched_gd, unmatched_pd = min_cost_matching2(costmatrix, gt_tracks2, estimated_tracks2)
        for match in matches:
            # 平均误差
            cost.append(costmatrix[match[0], match[1]])
            costRMSE.append(costmatrixRMSE[match[0], match[1]])

    if len(cost) != 0:
        meanP = round(np.mean(cost), 4)
        # MOTP = np.mean(cost)
        print('meanP=', meanP)
        all_meanP.append(meanP)

        n_targets = len(costRMSE)
        RMSE = round(np.sqrt(np.sum(costRMSE) / n_targets), 4)
        all_RMSE.append(RMSE)
        print('RMSE=', RMSE)
    # print('number of gt', groundtruth_number)
    Pre.append(TP / (TP + FP))
    Rec.append(TP / (TP + FN))

def run():


    all_FmeanP = []
    all_RmeanP = []
    all_CmeanP = []
    all_FRMSE = []
    all_RRMSE = []
    all_CRMSE = []
    allF_P = []  # TP, FP, FN
    allF_R = []
    allR_P = []
    allR_R = []
    allC_P = []
    allC_R = []

    max_distance = 6.0
    plot = False
    plot2 = False


    all_fusion_detections = np.load('./TrackResults/all_fusion_detections.npy')  # tracking result, (id,x,y,o, frame_num, group_num) 6
    all_gt_detections = np.load('./TrackResults/all_gt_detections.npy')  # gts, (id,x,y,z, classes, frame_num, group_num) 7
    all_radar_detections = np.load('./TrackResults/all_radar_detections_RMSE.npy')  # radar dets, (x,y, frame_num, group_num) 4
    all_img_detections = np.load('./TrackResults/all_img_detections_RMSE.npy')  # img dets, (x,y, frame_num, group_num) 4
    group_names = np.load('./TrackResults/group_names.npy')

    len_groups = int(max(all_gt_detections[:, 6]))

    for num_group in range(len_groups+1):
        print('num_group', num_group+1)
        print('group_name', group_names[num_group])
        group_name = group_names[num_group]
        # if num_group == 103:
        #     # print('num_group', num_group)
        #     print('*************'*10)
            # continue
        mask0 = np.ones(all_fusion_detections.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_fusion_detections[:, 3] == num_group)
        fusion_detections = all_fusion_detections[mask0, :]

        mask0 = np.ones(all_gt_detections.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_gt_detections[:, 6] == num_group)
        gt_detections = all_gt_detections[mask0, :]

        mask0 = np.ones(all_radar_detections.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_radar_detections[:, 3] == num_group)
        radar_detections = all_radar_detections[mask0, :]

        mask0 = np.ones(all_img_detections.shape[0], dtype=bool)
        mask0 = np.logical_and(mask0, all_img_detections[:, 3] == num_group)
        img_detections = all_img_detections[mask0, :]

        # filter
        # estimated_tracks = estimated_tracks[397:, :]
        # gt_tracks = gt_tracks[397:, :]
        # radar_detections = radar_detections[397:, :]
        # img_detections = img_detections[397:, :]

        if fusion_detections.shape[0] == 0 or gt_detections.shape[0] == 0 or radar_detections.shape[0] == 0 or img_detections.shape[0] == 0:
            continue

        num_min = int(min(gt_detections[:, 5]))
        num_max = int(max(gt_detections[:, 5]))
        print('length_lidar_frame:', num_max - num_min)

        # calculate FP and FN(miss)
        # costF = []
        # costFRMSE = []
        # costR = []
        # costRRMSE = []
        # costC = []
        # costCRMSE = []
        # groundtruth_number = 0

        # all_FmeanP = []
        # all_RmeanP = []
        # all_CmeanP = []
        # all_FRMSE = []
        # all_RRMSE = []
        # all_CRMSE = []

        # confusion = [0, 0, 0] # TP, FP, FN



        getMOTPandRMSE(gt_detections, fusion_detections, all_FmeanP, all_FRMSE, allF_P, allF_R)
        getMOTPandRMSE(gt_detections, radar_detections, all_CmeanP, all_CRMSE, allR_P, allR_R)
        getMOTPandRMSE(gt_detections, img_detections, all_RmeanP, all_RRMSE, allC_P, allC_R)




        # plt.xlim(-10, 10)
        # plt.ylim(0, 50)
        # title_name = str(int(group_name[0])) + '_' + group_name[1]
        # plt.title(title_name)
        # draw radar_detections, img_detections
        if plot:
            for i in range(100):
                mask_radar = np.ones(radar_detections.shape[0], dtype=bool)
                mask_radar = np.logical_and(mask_radar, radar_detections[:, 2] == i)
                radar_detections3 = radar_detections[mask_radar, :]

                mask_img = np.ones(img_detections.shape[0], dtype=bool)
                mask_img = np.logical_and(mask_img, img_detections[:, 2] == i)
                img_detections3 = img_detections[mask_img, :]

                # for i1 in range(radar_detections3.shape[0]):
                #     point = radar_detections3[i1, :]
                #     # random = np.random.RandomState(int(point[2]) + 13)
                #     # color = random.uniform(0., 1., size=3)
                #     color = [1.0, 0, 0]
                #     plt.scatter(point[0], point[1], color=color, marker='o', s=5, alpha=1)
                plt.scatter(radar_detections3[:, 0], radar_detections3[:, 1], color='b', marker='o', s=10, alpha=0.5)
                # for i2 in range(img_detections3.shape[0]):
                #     point = img_detections3[i2, :]
                #     # random = np.random.RandomState(int(point[2]) + 10)
                #     # color = random.uniform(0., 1., size=3)
                #     color = [0, 1.0, 0]
                #     plt.scatter(point[0], point[1], color=color, marker='x', s=5, alpha=1)
                plt.scatter(img_detections3[:, 0], img_detections3[:, 1], color='g', marker='x', s=10, alpha=0.5)
                # plt.pause(0.1)
                # plt.clf()
        # plt.show()

        # estimated_tracks, gt_tracks
        if plot2:
            # plot trajectory
            ids = set(estimated_tracks[:, 0])   # all estimate id
            # mark_a = (pd[:, 3] < 100.)
            # mark_a = np.logical_and(pd[:, 3] > 111., pd[:, 3] < 111. + 68.)
            # pd = pd[mark_a, :]
            # ii = 0
            # plt.figure(figsize=(3.348, 4.371))
            for i in ids:
                mask_pd = np.ones(estimated_tracks.shape[0], dtype=bool)
                mask_pd = np.logical_and(mask_pd, estimated_tracks[:, 0] == i)
                estimated_tracks_mask = estimated_tracks[mask_pd, :]
                # if ii == 0:
                #     color = 'b'
                # elif ii == 1:
                #     color = 'r'
                # elif ii == 2:
                #     color = 'y'
                # elif ii == 3:
                #     color = 'orange'
                # else:
                #     color = 'g'
                # ii = ii + 1
                plt.plot(estimated_tracks_mask[:, 1], estimated_tracks_mask[:, 2], color='r', alpha=1)

            # mark_a = (gd[:, 3] > lidar_frame)
            # gd = gd[mark_a, :]
            # mark_a = (gd[:, 3] < lidar_frame+100)
            # gd = gd[mark_a, :]
            ids = set(gt_tracks[:, 0])
            for j in ids:
                mask_gd = np.ones(gt_tracks.shape[0], dtype=bool)
                mask_gd = np.logical_and(mask_gd, gt_tracks[:, 0] == j)
                gt_tracks_mask = gt_tracks[mask_gd, :]
                plt.plot(gt_tracks_mask[:, 1], gt_tracks_mask[:, 2], color='k', linestyle='--', alpha=0.8)
                # gt_line, = plt.plot(gt_tracks_mask[:, 1], gt_tracks_mask[:, 2], color='k', linestyle='--', alpha=1)

            # ===================== draw Camera and Radar points
            # pd_CamPoints
            # pd_RadarPoints

            # pd = pd_predict
            # ids = set(pd[:, 0])
            # mark_a = (pd_CamPoints[:, 2] > 110.)
            # pd_CamPoints = pd_CamPoints[mark_a, :]
            # mark_a = (pd_CamPoints[:, 2] < 180.)
            # pd_CamPoints = pd_CamPoints[mark_a, :]
            # CamPoints = plt.scatter(pd_CamPoints[:, 0], pd_CamPoints[:, 1], color='g', s=2)
            # # plt.legend(['Visual points'], loc='lower right', fontsize=8)
            #
            # mark_a = (pd_RadarPoints[:, 2] > 110.)
            # pd_RadarPoints = pd_RadarPoints[mark_a, :]
            # mark_a = (pd_RadarPoints[:, 2] < 180.)
            # pd_RadarPoints = pd_RadarPoints[mark_a, :]
            # RadarPoints = plt.scatter(pd_RadarPoints[:, 0], pd_RadarPoints[:, 1], color='b', s=2)
            # # plt.legend(['Radar points'], loc='lower right', fontsize=8)
            #
            # # pd = pd_predict
            # # # ids = set(pd[:, 0])
            # # mark_a = (pd[:, 2] > 110.)
            # # pd = pd[mark_a, :]
            # # mark_a = (pd[:, 2] < 180.)
            # # pd = pd[mark_a, :]
            # # ii = 0
            # # # for i in ids:
            # # #     mask_pd = np.ones(pd.shape[0], dtype=bool)
            # # #     mask_pd = np.logical_and(mask_pd, pd[:, 0] == i)
            # # #     pd_mark = pd[mask_pd, :]
            # # if ii == 0:
            # #     color = 'b'
            # # elif ii == 1:
            # #     color = 'r'
            # # elif ii == 2:
            # #     color = 'y'
            # # elif ii == 3:
            # #     color = 'orange'
            # # else:
            # #     color = 'g'
            # # ii = ii + 1
            # # plt.scatter(pd[:, 0], pd[:, 1], color='r')
            #
            # # mark_a = (gd[:, 3] > lidar_frame)
            # # gd = gd[mark_a, :]
            # # mark_a = (gd[:, 3] < lidar_frame + 100)
            # # gd = gd[mark_a, :]
            # ids = set(gd[:, 0])
            #
            # # ================== end ==================
            #
            # plt.grid(True)
            # # plt.xlim(-6, 2)
            # # plt.ylim(2, 12)
            # plt.xlim(-3, 2)
            # plt.ylim(4.5, 11.5)
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # # plt.axis('scaled')
            # black_line = mlines.Line2D([], [], color='black', marker='_', linestyle='--',
            #                            markersize=5, label='Ground Truth')
            # green_line = mlines.Line2D([], [], color='g', marker='_',
            #                            markersize=5, label='Only Camera')
            # blue_line = mlines.Line2D([], [], color='b', marker='_',
            #                           markersize=5, label='Only Radar')
            # cyan_line = mlines.Line2D([], [], color='c', marker='_',
            #                           markersize=5, label='Tracking Trajectories Ⅰ')
            # orange_line = mlines.Line2D([], [], color='orange', marker='_',
            #                             markersize=5, label='Tracking Trajectories Ⅱ')
            # red_line = mlines.Line2D([], [], color='r', marker='_',
            #                          markersize=5, label='Proposed Algorithm')
            #
            # # plt.legend(loc='lower right', handles=[green_line, black_line], fontsize=10)
            # # plt.legend(loc='lower right', handles=[black_line], fontsize=8)
            # # plt.legend(['Visual points', 'Radar points', 'Ground Truth'], loc='lower right', fontsize=8)
            # plt.legend([CamPoints, RadarPoints, gt_line], ['Visual points', 'Radar points', 'Ground Truth'],
            #            loc='lower right', fontsize=10)
            #
            # # plt.title('fusion')
            # # plt.savefig('./fusion.pdf', bbox_inches='tight')
            #
            # plt.savefig('./Fig5a.png', bbox_inches='tight', dpi=1000)
            # plt.show()


        result_name = str(int(group_name[0])) + '_' + group_name[1] + '.png'
        SaveRawPath = '/mnt/ChillDisk/personal_data/zhangq/RadarRGBFusionNet2_20231128/TrackResults/Figs'
        save_dir = os.path.join(SaveRawPath, result_name)
        # plt.savefig(save_dir, dpi=800)
        # plt.show()
        plt.clf()


        print('***********************'*5, '\n')


    print('all_FmeanP', np.mean(all_FmeanP))
    print('all_FRMSE', np.mean(all_FRMSE))
    print('all_RmeanP', np.mean(all_RmeanP))
    print('all_RRMSE', np.mean(all_RRMSE))
    print('all_CmeanP', np.mean(all_CmeanP))
    print('all_CRMSE', np.mean(all_CRMSE))
    print('F_Pre: ', np.mean(allF_P))
    print('F_Recall: ', np.mean(allF_R))
    print('R_Pre: ', np.mean(allR_P))
    print('R_Recall: ', np.mean(allR_R))
    print('C_Pre: ', np.mean(allC_P))
    print('C_Recall: ', np.mean(allC_R))
    print('\ndone')

if __name__ == '__main__':
    run()