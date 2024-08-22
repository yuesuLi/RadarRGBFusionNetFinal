# vim: expandtab:ts=4:sw=4
#-*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """
    Computer intersection over union.（计算iou）
    ==========================================================================
    Parameters
    ----------
    bbox : ndarray
        tlwh形式的bounding box
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates(候选框) : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
        第一维是candidates的索引，每个candidates下是和bbox相同格式
    ===========================================================================
    Returns
    -------
    iou : ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
        取值在[0,1]间，第一维是Index,第二维是iou

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    # top left和bottom right
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """
    An intersection over union distance metric.
    iou距离尺度
    =======================================================================
    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.（列表形式的tracks）
    detections : List[deep_sort.detection.Detection]
        A list of detections.（列表形式的Detection）
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.(列表形式的tracks的索引，id列表？)
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.（列表形式的Detection的索引）
    ========================================================================
    Returns
    -------
    cost_matrix : ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
        返回维度为len(track_indices), len(detection_indices) 的矩阵
        每个元素（i,j）代表第i个tracks和第j个detection的iou_cost
        iou_cost是用1-iou算出来的，iou越小，说明两者距离越远，iou_cost越大
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    # 如果没有track和detection的索引，则根据tracks和detections的长度创建

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        # track_indices包含的是各个track的索引即id，enumerate给track_indices中每个元素又分配了一个索引
        if tracks[track_idx].time_since_update > 1:
            # 如果id为track_idx的tracks的time_since_update>1
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            # ？
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
