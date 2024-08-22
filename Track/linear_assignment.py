# vim: expandtab:ts=4:sw=4
#-*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment
from Track import nn_matching
from copy import deepcopy
# from .weighted_graph import WeightedGraph
from . import kalman_filter, tracker
from scipy.spatial import distance
import time

INFTY_COST = 1e+5


def iou(bbox, candidates):
    """
    Computer intersection over union.
    ==========================================================================
    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
    ===========================================================================
    Returns
    -------
    iou : ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    # top left和bottom right
    candidates_tl = candidates[:2]
    candidates_br = candidates[:2] + candidates[2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[0]),
               np.maximum(bbox_tl[1], candidates_tl[1])]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[0]),
               np.minimum(bbox_br[1], candidates_br[1])]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod()
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[2:].prod()
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def speed_direction_batch(dets, tracks_pre_obs):
    """
    input:
        dets: (num_dets, 2), (x, y)
        tracks_pre_obs: tracks' observation, (N_track, 3) (x, y, c)

    return: norm_dy, norm_dx (num_track, num_det)

    """
    tracks_pre_obs = tracks_pre_obs[..., np.newaxis]
    X1, Y1 = dets[:, 0], dets[:, 1]
    X2, Y2 = tracks_pre_obs[:, 0], tracks_pre_obs[:, 1]
    dx = X1 - X2
    dy = Y1 - Y2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

def min_cost_matching(
        cost_matrix, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.
    there can set max distance, distance > max_distance assignement can be cancel
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    cost_matrix = cost_matrix[track_indices, :]
    cost_matrix = cost_matrix[:, detection_indices]
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e5
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

def my_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def matching_detections(detections_r, detections_c, detection_r_indices=None, detection_c_indices=None):

    if detection_r_indices is None:
        detection_r_indices = np.arange(len(detections_r))
    if detection_c_indices is None:
        detection_c_indices = np.arange(len(detections_c))

    # distance_matrix_r: (num_tracks, num_det_r);   distance_matrix_c: (num_tracks, num_det_c)
    detections_costmatrix = get_detections_cost_matrix(detections_r, detections_c, detection_r_indices,
                                                                      detection_c_indices)
    # distance_threshold: 这个阈值表示目标之间的距离必须要小于这个值，不然不会被关联
    distance_threshold1 = 1.5
    if min(detections_costmatrix.shape) > 0:
        a = (detections_costmatrix <= distance_threshold1).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = my_assignment(detections_costmatrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections_r = []
    for d_r, det_r in enumerate(detections_r):
        if (d_r not in matched_indices[:, 0]):
            unmatched_detections_r.append(d_r)
    unmatched_detections_c = []
    for d_c, det_c in enumerate(detections_c):
        if (d_c not in matched_indices[:, 1]):
            unmatched_detections_c.append(d_c)

    # filter out matched with far distance
    distance_threshold2 = 1.5
    matches = []
    for m in matched_indices:
        if (detections_costmatrix[m[0], m[1]] > distance_threshold2):
            unmatched_detections_r.append(m[0])
            unmatched_detections_c.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections_r), np.array(unmatched_detections_c)


def matching_cascade(tracks, AllDetections, velocities, previous_obs, track_indices=None, vdc_weight=5, app_weight=5):
    """
    input:
        tracks: A list of tracks
        AllDetections: A list of detections
        velocities: (N_track, 2)  (norm_dy, norm_dx)
        previous_obs: (N_track, 3) (x, y, c)
    -------
    return:
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """

    detections_indices = np.arange(len(AllDetections))
    if len(track_indices) == 0:
        return np.empty((0, 2), dtype=int), [], list(detections_indices)
    if len(detections_indices) == 0:
        return np.empty((0, 2), dtype=int), list(track_indices), list(detections_indices)


    # if (len(track_indices) == 0):
    #     return np.empty((0, 2), dtype=int), np.empty((0, 3), dtype=int),  np.arange(len(AllDetections))

    # distance_costmatrix: (num_tracks, num_det_r), smaller better
    distance_costmatrix = get_distance_cost_matrix(tracks, AllDetections, track_indices, detections_indices)
    # pos_gate = distance_costmatrix > 1.0

    # orientation_costmatrix: (num_tracks, num_det_r), smaller better
    orientation_costmatrix = get_ori_cost_matrix(tracks, AllDetections, velocities, previous_obs, vdc_weight, track_indices, detections_indices)
    # orientation_costmatrix = get_ori_cost_matrix2(tracks, AllDetections, velocities, previous_obs, vdc_weight, track_indices, detections_indices)
    orientation_costmatrix = orientation_costmatrix * vdc_weight
    #
    appearance_costmatrix = get_appearance_cost_matrix(tracks, AllDetections, app_weight, track_indices, detections_indices)
    appearance_costmatrix = appearance_costmatrix * app_weight
    # app_gate = app_cost > self.metric.matching_threshold
    # cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
    # cost_matrix[np.logical_or(pos_gate, app_gate)] = INFTY_COST

    # all_costmatrix = distance_costmatrix
    all_costmatrix = distance_costmatrix + orientation_costmatrix + appearance_costmatrix


    # print('distance_costmatrix: ', distance_costmatrix.shape, '\n', distance_costmatrix)
    # print('orientation_costmatrix: ', orientation_costmatrix.shape, '\n', orientation_costmatrix * vdc_weight)
    # print('appearance_costmatrix: ', appearance_costmatrix.shape, '\n', appearance_costmatrix * app_weight)


    # distance_threshold: 这个阈值表示目标之间的距离必须要小于这个值，不然不会被关联
    distance_threshold1 = 1.5
    if min(distance_costmatrix.shape) > 0:
        a = (distance_costmatrix <= distance_threshold1).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = my_assignment(all_costmatrix)
    else:
        matched_indices = np.empty(shape=(0, 2))


    unmatched_tracks = []
    for t, trk in enumerate(tracks):
        if (t not in matched_indices[:, 0]):
            unmatched_tracks.append(t)

    unmatched_detections = []
    for d, det in enumerate(AllDetections):
        if (d not in matched_indices[:, 1]):
            unmatched_detections.append(d)


    matches = []

    # filter out matched with far distance
    distance_threshold2 = 2.0
    for m in matched_indices:
        if(distance_costmatrix[m[0], m[1]] > distance_threshold2):
            unmatched_tracks.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches)==0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return np.array(matches), unmatched_tracks, unmatched_detections

    # return matches, unmatched_tracks, unmatched_detections


def Assignment(track_indices, detections_indices, distance_costmatrix, all_costmatrix, fusionState):
    # distance_threshold: 这个阈值表示目标之间的距离必须要小于这个值，不然不会被关联
    if fusionState == 3:
        distance_threshold = 2.0
        # filter out matched with far distance
        distance_threshold2 = 2.5
    elif fusionState == 1:
        distance_threshold = 1.5
        distance_threshold2 = 2.0
    elif fusionState == 2:
        distance_threshold = 1.0
        distance_threshold2 = 1.5

    if min(distance_costmatrix.shape) > 0:
        a = (distance_costmatrix <= distance_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = my_assignment(all_costmatrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # true_matched_indices = deepcopy(matched_indices)
    # for i in range(len(matched_indices)):
    #     true_matched_indices[i][0] = track_indices[matched_indices[i][0]]

    unmatched_tracks = []
    for t, trackIndex in enumerate(track_indices):
        if (t not in matched_indices[:, 0]):
            unmatched_tracks.append(trackIndex)

    unmatched_detections = []
    for d, detIndex in enumerate(detections_indices):
        if (d not in matched_indices[:, 1]):
            unmatched_detections.append(detIndex)

    matches = []
    for m in matched_indices:
        if (distance_costmatrix[m[0], m[1]] > distance_threshold2):
            unmatched_tracks.append(track_indices[m[0]])
            unmatched_detections.append(detections_indices[m[1]])
        else:
            matches.append(np.array([track_indices[m[0]], detections_indices[m[1]]]).reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, unmatched_tracks, unmatched_detections  # np.array, list, list

def getCostmatrix(tracks, detections, track_indices, detections_indices, velocities, previous_obs, vdc_weight, app_weight, fusionState):
    """
    fusion_state:
        0: no target
        1: only_camera
        2: only_radar
        3: fusion_sensors
    """
    if len(detections_indices) == 0 or len(track_indices) == 0:
        return np.empty((0, 2), dtype=int), track_indices, list(detections_indices)


    if fusionState == 3:
        # distance_costmatrix: (num_tracks, num_det_r), smaller better
        distance_costmatrix = get_distance_cost_matrix(tracks, detections, track_indices, detections_indices)

        # orientation_costmatrix: (num_tracks, num_det_r), smaller better, [-0.5, 0.5], default = 0
        orientation_costmatrix = get_ori_cost_matrix(tracks, detections, velocities, previous_obs, vdc_weight,
                                                     track_indices, detections_indices)
        orientation_costmatrix = orientation_costmatrix * vdc_weight

        # appearance_costmatrix: (num_tracks, num_det_r), smaller better [-0.5, 0.5], default = 0
        appearance_costmatrix = get_appearance_cost_matrix(tracks, detections, app_weight, track_indices,
                                                           detections_indices)
        appearance_costmatrix = appearance_costmatrix * app_weight
        # app_gate = app_cost > self.metric.matching_threshold
        # cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
        # cost_matrix[np.logical_or(pos_gate, app_gate)] = INFTY_COST

        # all_costmatrix = distance_costmatrix
        all_costmatrix = distance_costmatrix + appearance_costmatrix + orientation_costmatrix
        matches, unmatched_tracks, unmatched_detections = Assignment(track_indices, detections_indices, distance_costmatrix, all_costmatrix, fusionState)

        return matches, unmatched_tracks, unmatched_detections
    elif fusionState == 1:
        # distance_costmatrix: (num_tracks, num_det_r), smaller better
        distance_costmatrix = get_distance_cost_matrix(tracks, detections, track_indices, detections_indices)

        # orientation_costmatrix: (num_tracks, num_det_r), smaller better, [-0.5, 0.5], default = 0
        orientation_costmatrix = get_ori_cost_matrix(tracks, detections, velocities, previous_obs, vdc_weight,
                                                     track_indices, detections_indices)
        orientation_costmatrix = orientation_costmatrix * vdc_weight

        # appearance_costmatrix: (num_tracks, num_det_r), smaller better [-0.5, 0.5], default = 0
        appearance_costmatrix = get_appearance_cost_matrix(tracks, detections, app_weight, track_indices,
                                                           detections_indices)
        appearance_costmatrix = appearance_costmatrix * app_weight
        # app_gate = app_cost > self.metric.matching_threshold
        # cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
        # cost_matrix[np.logical_or(pos_gate, app_gate)] = INFTY_COST

        # all_costmatrix = distance_costmatrix
        all_costmatrix = distance_costmatrix + appearance_costmatrix + orientation_costmatrix
        matches, unmatched_tracks, unmatched_detections = Assignment(track_indices, detections_indices,
                                                                     distance_costmatrix, all_costmatrix, fusionState)

        return matches, unmatched_tracks, unmatched_detections


    elif fusionState == 2:
        # distance_costmatrix: (num_tracks, num_det_r), smaller better
        distance_costmatrix = get_distance_cost_matrix(tracks, detections, track_indices, detections_indices)

        # orientation_costmatrix: (num_tracks, num_det_r), smaller better, [-0.5, 0.5], default = 0
        orientation_costmatrix = get_ori_cost_matrix(tracks, detections, velocities, previous_obs, vdc_weight,
                                                     track_indices, detections_indices)
        orientation_costmatrix = orientation_costmatrix * vdc_weight

        # all_costmatrix = distance_costmatrix
        all_costmatrix = distance_costmatrix + orientation_costmatrix
        matches, unmatched_tracks, unmatched_detections = Assignment(track_indices, detections_indices,
                                                                     distance_costmatrix, all_costmatrix, fusionState)

        return matches, unmatched_tracks, unmatched_detections


def AssignDisCostmatrix(tracks, detections, track_indices, DetectionIndexes, app_weight, fusionState):

    if len(track_indices) == 0 or len(DetectionIndexes) == 0:
        return np.empty((0, 2), dtype=int), track_indices, DetectionIndexes



    # distance_costmatrix: (num_tracks, num_det_r), smaller better
    distance_costmatrix = get_distance_cost_matrix(tracks, detections, track_indices, DetectionIndexes)

    # appearance_costmatrix: (num_tracks, num_det_r), smaller better [-0.5, 0.5], default = 0
    appearance_costmatrix = get_appearance_cost_matrix(tracks, detections, app_weight, track_indices,
                                                       DetectionIndexes)
    appearance_costmatrix = appearance_costmatrix * app_weight
    all_costmatrix = distance_costmatrix + appearance_costmatrix
    matches, unmatched_tracks, unmatched_detections = Assignment(track_indices, DetectionIndexes,
                                                                 distance_costmatrix, all_costmatrix, fusionState)

    return matches, unmatched_tracks, unmatched_detections

def matching_cascade_threeStage(tracks, FusionDetections, CameraDetections, RadarDetections, velocities, previous_obs, track_indices, vdc_weight=5, app_weight=5):
    """
    input:
        tracks: A list of tracks
        AllDetections: A list of detections
        velocities: (N_track, 2)  (norm_dy, norm_dx)
        previous_obs: (N_track, 3) (x, y, c)

        fusion_state:
        0: no target
        1: only_camera
        2: only_radar
        3: fusion_sensors
    -------
    return:
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """

    FusionDetections_indices = np.arange(len(FusionDetections))
    CameraDetections_indices = np.arange(len(CameraDetections))
    RadarDetections_indices = np.arange(len(RadarDetections))


    if len(track_indices) == 0 or len(FusionDetections) + len(CameraDetections) + len(RadarDetections) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int),
                track_indices, list(FusionDetections_indices), list(CameraDetections_indices), list(RadarDetections_indices))


    # track_indices_l = [
    #     k for k in track_indices
    #     if tracks[k].assigned == 0
    # ]

    matchesF, unmatched_tracks, unmatched_FusionDetections = getCostmatrix(tracks, FusionDetections, track_indices,
                                                                    FusionDetections_indices, velocities, previous_obs,
                                                                    vdc_weight, app_weight, fusionState=3)

    # for assignTrack in matchesF[:, 0]:
    #     tracks[assignTrack].assigned == 1
    # track_indices_l = [
    #     k for k in track_indices
    #     if tracks[k].assigned == 0
    # ]


    matchesC, unmatched_tracks, unmatched_CameraDetections = getCostmatrix(tracks, CameraDetections, unmatched_tracks,
                                                                          CameraDetections_indices, velocities,
                                                                          previous_obs,
                                                                          vdc_weight, app_weight, fusionState=1)



    matchesR, unmatched_tracks, unmatched_RadarDetections = getCostmatrix(tracks, RadarDetections, unmatched_tracks,
                                                                           RadarDetections_indices, velocities,
                                                                           previous_obs,
                                                                           vdc_weight, app_weight, fusionState=2)
    # matches = np.concatenate((matchesF, matchesC, matchesR), axis=0)


    return matchesF, matchesC, matchesR, unmatched_tracks, unmatched_FusionDetections, unmatched_CameraDetections, unmatched_RadarDetections

    # return matches, unmatched_tracks, unmatched_detections


def matching_init(tracks, FusionDetections, CameraDetections, track_indices, FusionIndexes, CameraIndexes, app_weight=5):

    if len(track_indices) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int),
                track_indices, FusionIndexes, CameraIndexes)
    if len(FusionIndexes) + len(CameraIndexes) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int),
                track_indices, FusionIndexes, CameraIndexes)


    matchesF, unmatched_tracks, unmatched_FusionDetections = AssignDisCostmatrix(tracks, FusionDetections, track_indices,
                                                                    FusionIndexes, app_weight, fusionState=3)

    matchesC, unmatched_tracks, unmatched_CameraDetections = AssignDisCostmatrix(tracks, CameraDetections, unmatched_tracks,
                                                                           CameraIndexes, app_weight, fusionState=1)


    return matchesF, matchesC, unmatched_tracks, unmatched_FusionDetections, unmatched_CameraDetections

def second_position_match(tracks, AllDetections, track_indices=None, detections_indices=None):

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detections_indices is None:
        detections_indices = np.arange(len(AllDetections))

    # if (len(track_indices) == 0):
    #     return np.empty((0, 2), dtype=int), np.empty((0, 3), dtype=int),  np.arange(len(AllDetections))

    # distance_costmatrix: (num_tracks, num_det_r), smaller better
    distance_costmatrix = get_distance_cost_matrix(tracks, AllDetections, track_indices, detections_indices)
    # distance_costmatrix[distance_costmatrix > max_distance] = max_distance + 1e-5
    all_costmatrix = distance_costmatrix

    # distance_threshold: 这个阈值表示目标之间的距离必须要小于这个值，不然不会被关联
    distance_threshold1 = 3.5
    if min(distance_costmatrix.shape) > 0:
        a = (distance_costmatrix <= distance_threshold1).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = my_assignment(all_costmatrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_tracks = []
    for t, trk in enumerate(tracks):
        if (t not in matched_indices[:, 0]):
            unmatched_tracks.append(t)

    unmatched_detections = []
    for d, det in enumerate(AllDetections):
        if (d not in matched_indices[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # filter out matched with far distance
    distance_threshold2 = 5.0
    for m in matched_indices:
        if (distance_costmatrix[m[0], m[1]] > distance_threshold2):
            unmatched_tracks.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return np.array(matches), unmatched_tracks, unmatched_detections

def get_detections_cost_matrix(
        detections_r, detections_c, detection_r_indices=None,
        detection_c_indices=None, gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    cost_matrix = np.ones((len(detections_r), len(detections_c)))*100
    if detection_r_indices is None:
        detection_r_indices = np.arange(len(detections_r))
    if detection_c_indices is None:
        detection_c_indices = np.arange(len(detections_c))

    measurement_r = np.asarray(
        [detections_r[i].center[0:2] for i in detection_r_indices])
    measurement_c = np.asarray(
        [detections_c[i].center[0:2] for i in detection_c_indices])

    for i in range(measurement_r.shape[0]):
        for j in range(measurement_c.shape[0]):
            cost_matrix[i, j] = np.sqrt((measurement_r[i, 0] - measurement_c[j, 0]) ** 2 + \
                                        (measurement_r[i, 1] - measurement_c[j, 1]) ** 2)

    return cost_matrix

def get_distance_cost_matrix(tracks, AllDetections, track_indices=None, detections_indices=None,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    cost_matrix : ndarray
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """


    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detections_indices is None:
        detections_indices = np.arange(len(AllDetections))

    distance_costmatrix = np.ones((len(track_indices), len(detections_indices))) * 1e6
    # distance_costmatrix_x = np.ones((len(track_indices), len(detections_indices))) * 100
    # distance_costmatrix_y = np.ones((len(track_indices), len(detections_indices))) * 100

    track_position = np.asarray(
        [tracks[i].mean[0:2] for i in track_indices])
    all_measurements = np.asarray(
        [AllDetections[i].center[0:2] for i in detections_indices])

    for i in range(track_position.shape[0]):
        for j in range(all_measurements.shape[0]):
            distance_costmatrix[i, j] = np.sqrt((track_position[i, 0] - all_measurements[j, 0]) ** 2 +
                                                (track_position[i, 1] - all_measurements[j, 1]) ** 2)
            # distance_costmatrix_x[i, j] = np.sqrt((track_position[i, 0] - all_measurements[j, 0]) ** 2)
            # distance_costmatrix_y[i, j] = np.sqrt((track_position[i, 1] - all_measurements[j, 1]) ** 2)

    # distance_costmatrix = distance_costmatrix_x + distance_costmatrix_y
    # print('distance_costmatrix:\n', distance_costmatrix)
    # print('distance_costmatrix_x:\n', distance_costmatrix_x)
    # print('distance_costmatrix_y:\n', distance_costmatrix_y)
    # print('distance_costmatrix2:\n', distance_costmatrix_x + distance_costmatrix_y, '\n')

    return distance_costmatrix  # (num_track, num_det)


def get_appearance_cost_matrix(tracks, AllDetections, app_weight=5, track_indices=None,
                        detections_indices=None, gated_cost=INFTY_COST):
    """
    input:
        tracks: A list of tracks
        AllDetections: A list of detections
            for detections, fusion_state:
                0: no target
                1: only_camera
                2: only_radar
                3: fusion_sensors
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detections_indices is None:
        detections_indices = np.arange(len(AllDetections))

    tracks_feature = np.asarray(
        [tracks[i].feature for i in track_indices])  # (num_tracks, 1, 512)
    tracks_feature = tracks_feature.reshape(tracks_feature.shape[0], tracks_feature.shape[2])   # (num_tracks, 512)

    detections_measurement = np.asarray(
        [AllDetections[i].feature for i in detections_indices])  # (num_dets, 1, 512)
    detections_measurement = detections_measurement.reshape(detections_measurement.shape[0],
                                                            detections_measurement.shape[2])    # (num_dets, 512)

    app_cost_class = nn_matching.NearestNeighborDistanceMetric(metric="cosine", matching_threshold=0.3)
    appearance_costmatrix = app_cost_class.distance(tracks_feature, detections_measurement)

    # AllDetections_state = np.asarray(
    #     [AllDetections[i].fusion_state for i in detections_indices])  # (num_dets, 1)
    # appearance_mask = np.ones(detections_measurement.shape[0])  # (num_dets, )
    # appearance_mask[np.where(AllDetections_state == 2)] = 0
    # appearance_mask[np.where(AllDetections_state == 0)] = 0
    # appearance_mask = np.repeat(appearance_mask[np.newaxis, :], appearance_costmatrix.shape[0], axis=0)  # (num_track, num_det)
    # appearance_costmatrix = appearance_mask * appearance_costmatrix  # (num_track, num_det)

    return appearance_costmatrix
def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-100, -100, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]    # (x, y, c)

def get_ori_cost_matrix(tracks, AllDetections, velocities, previous_obs, vdc_weight=5, track_indices=None,
                        detections_indices=None, gated_cost=INFTY_COST):
    """
    input:
        tracks: A list of tracks
        AllDetections: A list of detections
        velocities: tracks' velocities, (N_track, 2)  (norm_dy, norm_dx)
        previous_obs: tracks' observation, (N_track, 3) (x, y, c)
                      if no obs , previous_obs=[-100, -100, -1]
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detections_indices is None:
        detections_indices = np.arange(len(AllDetections))


    # orientation_tracks = np.asarray(
    #     [tracks[i].track_orientation for i in track_indices])   # (num_tracks, 2)
    detections_measurement = np.asarray(
        [AllDetections[i].center[0:2] for i in detections_indices])     # (num_dets, 2), (x, y)

    # previous_obs = np.asarray([previous_obs[i] for i in track_indices])
    # velocities = np.asarray([velocities[i] for i in track_indices])
    velocities = np.array(
        [tracks[i].velocity if tracks[i].velocity is not None else np.array((0, 0)) for i in
         track_indices])  # (N_track, 2)  (norm_dy, norm_dx)
    previous_obs = np.array(
        [k_previous_obs(tracks[i].observations, tracks[i].age, tracks[i].delta_t) for i in
         track_indices])  # (N_track, 3) (x, y, c)
    # print('detections_measurement', detections_measurement)
    Y, X = speed_direction_batch(detections_measurement, previous_obs)  # norm_dy, norm_dx (num_track, num_det)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi # (num_track, num_det)
    # diff_angle = np.abs(diff_angle) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 2] < 0)] = 0    # mask no observation
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)   # (num_track, num_det)

    angle_diff_cost = (valid_mask * diff_angle) * -1    # (num_track, num_det)

    return angle_diff_cost

def get_ori_cost_matrix2(tracks, AllDetections, velocities, previous_obs, vdc_weight=5, track_indices=None,
                        detections_indices=None, gated_cost=INFTY_COST):
    """
    input:
        tracks: A list of tracks
        AllDetections: A list of detections
        velocities: tracks' velocities, (N_track, 2)  (norm_dy, norm_dx)
        previous_obs: tracks' observation, (N_track, 3) (x, y, c)
                      if no obs , previous_obs=[-100, -100, -1]
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detections_indices is None:
        detections_indices = np.arange(len(AllDetections))


    orientation_tracks = np.asarray(
        [float(tracks[i].mean[2]) for i in track_indices])   # (num_tracks, )
    detections_measurement = np.asarray(
        [AllDetections[i].center[0:2] for i in detections_indices])     # (num_dets, 2), (x, y)
    # print('detections_measurement', detections_measurement)

    previous_obs = np.array(
        [k_previous_obs(tracks[i].observations, tracks[i].age, tracks[i].delta_t) for i in
         track_indices])  # (N_track, 3) (x, y, c)
    Y, X = speed_direction_batch(detections_measurement, previous_obs)  # norm_dy, norm_dx (num_track, num_det)
    detections_orientation = np.arctan2(Y, X)   # (num_track, num_det), track2dets' orientation, [-pi, pi]
    tracks_orientation = np.repeat(orientation_tracks[:, np.newaxis], Y.shape[1], axis=1)   # (num_track, num_det) tracks' orientation, [-pi, pi]

    diff_angle = np.abs(detections_orientation - tracks_orientation) # [-2pi, 2pi] ---> [0, 2pi], smaller is better
    diff_angle = (np.abs(diff_angle) - np.pi) / (2 * np.pi)  # (num_track, num_det)  [-0.5, 0.5], smaller is better
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 2] < 0)] = 0  # mask no observation
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # (num_track, num_det)
    angle_diff_cost = (valid_mask * diff_angle)  # (num_track, num_det), [-0.5, 0.5], smaller is better

    return angle_diff_cost

def get_tracks_costmatrix(tracks, track_indices=None):
    """
        function:
            get costmatrix between tracks

        input:
            tracks: A list of tracks, len:N

        return:
            costmatrix between tracks, (N, N)
        """

    tracks_num = len(tracks)
    if tracks_num <= 1:
        return np.array([])
    tracks_distance_costmatrix = np.ones((tracks_num, tracks_num)) * 1e6
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    track_position = np.asarray(
        [tracks[i].mean[0:2].astype('float') for i in track_indices])

    # print('track_position: ', track_position)
    for i in range(track_position.shape[0]):
        for j in range(track_position.shape[0]):
            if i != j:
                tracks_distance_costmatrix[i, j] = np.sqrt((track_position[i, 0] - track_position[j, 0]) ** 2 + \
                                                           (track_position[i, 1] - track_position[j, 1]) ** 2)

    return tracks_distance_costmatrix


def get_matched_indices(matched_indices_r, matched_indices_c):
    len_r = matched_indices_r.shape[0]
    len_c = matched_indices_c.shape[0]

    matched_indices = np.ones((len_r + len_c, 3)) * -1

    if len_r > 0 and len_c > 0:
        for i in range(len_r):
            for j in range(len_c):
                if matched_indices_r[i, 0] == matched_indices_c[j, 0]:
                    matched_indices[i] = [matched_indices_r[i, 0], matched_indices_r[i, 1], matched_indices_c[j, 1]]
                    break
                elif j == len_c - 1:
                    matched_indices[i] = [matched_indices_r[i, 0], matched_indices_r[i, 1], -1]

        idx = matched_indices.shape[0] - 1
        for i in range(len_c):
            for j in range(len_r):
                if matched_indices_c[i, 0] == matched_indices_r[j, 0]:
                    break
                elif j == len_r - 1:
                    matched_indices[idx] = [matched_indices_c[i, 0], -1, matched_indices_c[i, 1]]
                    idx -= 1
    elif len_r == 0:
        for i in range(len_c):
            matched_indices[i] = [matched_indices_c[i, 0], -1, matched_indices_c[i, 1]]
    elif len_c == 0:
        for i in range(len_r):
            matched_indices[i] = [matched_indices_r[i, 0], matched_indices_r[i, 1], -1]

    print('matched_indices:', matched_indices)
    return matched_indices


def __get_conflicting_tracks(track_detections):
    conflicting_tracks = []     # n=1, so prune from current frame, examine whether have conflict
    for i in range(len(track_detections)):
        for j in range(i + 1, len(track_detections)):   # from i+1 to len(track_detections)
            left_ids = track_detections[i]
            right_ids = track_detections[j]
            for k in range(len(left_ids)):
                if left_ids[k] != '' and right_ids[k] != '' and left_ids[k] == right_ids[k]:
                    conflicting_tracks.append((i, j))   # if left != right, conflict exist

    return conflicting_tracks


def __global_hypothesis(track_trees, conflicting_tracks):
    """
    Generate a global hypothesis by finding the maximum weighted independent
    set of a graph with tracks as vertices, and edges between conflicting tracks.
    """
    # create undirected graph and find the best result
    gh_graph = WeightedGraph()
    for index, score in enumerate(track_trees):
        gh_graph.add_weighted_vertex(str(index), score)

    gh_graph.set_edges(conflicting_tracks)

    mwis_ids = gh_graph.mwis()

    return mwis_ids