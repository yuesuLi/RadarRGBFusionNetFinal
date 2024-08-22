# vim: expandtab:ts=4:sw=4
# -*- coding:utf-8 -*-
from __future__ import absolute_import
import numpy as np
from .kalman_filter_fusion import KalmanFilter_fusion
from . import linear_assignment
from .track import Track



def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-100, -100, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]    # (x, y, c)

# def match_detections(detections_r, detections_c):
#     # Associate radar and img detections using position
#     matches, unmatched_detections_r, unmatched_detections_c\
#         = linear_assignment.matching_detections(detections_r, detections_c)
#
#     return matches, unmatched_detections_r, unmatched_detections_c

def get_all_detections(detections_r, detections_c):
    # Associate radar and img detections using position
    detection_r_indices = np.arange(len(detections_r))
    detection_c_indices = np.arange(len(detections_c))
    matches, unmatched_detections_r, unmatched_detections_c \
        = linear_assignment.matching_detections(detections_r, detections_c, detection_r_indices, detection_c_indices)

    all_detections = []

    for detection_r_idx, detection_c_idx in matches:
        detection_r = detections_r[int(detection_r_idx)]
        detection_c = detections_c[int(detection_c_idx)]

        center_r = detection_r.center
        center_c = detection_c.center
        detection_r.center = center_r * 0.9 + center_c * 0.1
        # print('center_r, center_c, fusion_center:\n', center_r, center_c, detection_r.center)
        detection_r.fusion_state = 3
        detection_r.feature = detection_c.feature
        detection_r.tlwh = detection_c.tlwh
        detection_r.classes = detection_c.classes
        detection_r.confidence = detection_c.confidence
        # self.tracks[int(track_idx)].update(self.kf, detection, self.kf_box)
        all_detections.append(detection_r)

    for detection_r1_idx in unmatched_detections_r:
        detection_r1 = detections_r[int(detection_r1_idx)]
        all_detections.append(detection_r1)

    for detection_c1_idx in unmatched_detections_c:
        detection_c1 = detections_c[int(detection_c1_idx)]
        all_detections.append(detection_c1)

    return all_detections

def get_split_detections(detections_r, detections_c):
    # Associate radar and img detections using position
    detection_r_indices = np.arange(len(detections_r))
    detection_c_indices = np.arange(len(detections_c))
    matches, unmatched_detections_r, unmatched_detections_c \
        = linear_assignment.matching_detections(detections_r, detections_c, detection_r_indices, detection_c_indices)

    fusion_detections, camera_detections, radar_detections = [], [], []

    for detection_r_idx, detection_c_idx in matches:
        detection_r = detections_r[int(detection_r_idx)]
        detection_c = detections_c[int(detection_c_idx)]

        center_r = detection_r.center
        center_c = detection_c.center
        detection_r.center = center_r * 0.5 + center_c * 0.5
        # print('center_r, center_c, fusion_center:\n', center_r, center_c, detection_r.center)
        detection_r.fusion_state = 3
        detection_r.feature = detection_c.feature
        detection_r.tlwh = detection_c.tlwh
        detection_r.classes = detection_c.classes
        detection_r.confidence = detection_c.confidence
        # self.tracks[int(track_idx)].update(self.kf, detection, self.kf_box)
        fusion_detections.append(detection_r)

    for detection_c1_idx in unmatched_detections_c:
        detection_c1 = detections_c[int(detection_c1_idx)]
        camera_detections.append(detection_c1)

    for detection_r1_idx in unmatched_detections_r:
        detection_r1 = detections_r[int(detection_r1_idx)]
        radar_detections.append(detection_r1)

    return fusion_detections, camera_detections, radar_detections


def calculate_intersection_point(track1, track2):
    """
        track.mean: (x, y, o, dx, dy, do)
        track.velocity: (norm_dy, norm_dx), (2,)

        return: (intersection_point_x, intersection_point_y)
    """

    track_position1 = track1.mean[0:2]
    track_position2 = track2.mean[0:2]
    track_orientation1 = np.divide(track1.velocity[0], track1.velocity[1] + 1e-6)
    track_orientation2 = np.divide(track2.velocity[0], track2.velocity[1] + 1e-6)

    x_numerator = track_orientation1 * track_position1[0] - track_orientation2 * track_position2[0]\
                  + track_position2[1] - track_position1[1]
    x_denominator = track_orientation1 - track_orientation2
    intersection_point_x = np.divide(x_numerator, x_denominator + 1e-6)

    y_numerator = track_orientation1 * track_orientation2 * (track_position1[0] - track_position2[0])\
                  + track_orientation1 * track_position2[1] - track_orientation2 * track_position1[1]
    y_denominator = track_orientation1 - track_orientation2
    intersection_point_y = np.divide(y_numerator, y_denominator + 1e-6)

    intersection_point = np.array([float(intersection_point_x), float(intersection_point_y)])

    return intersection_point


def getTrackDis(currTrack):

    curr_xy = [float(currTrack.mean[0]), float(currTrack.mean[1])]
    rel_xy = [float(currTrack.related_track.mean[0]), float(currTrack.related_track.mean[1])]
    dis = np.sqrt((curr_xy[0] - rel_xy[0]) ** 2 + (curr_xy[1] - rel_xy[1]) ** 2)

    return dis


class Tracker:

    def __init__(self, max_age=3, n_init=3, z=None, R=None, T=None, V=None, focal=None, W=None, H=None):
        self.max_age = max_age
        self.n_init = n_init
        # self.kf = KalmanFilter_fusion.KalmanFilter()
        # self.kf_box = kalman_filter_cam.KalmanFilter()
        self.tracks = []
        # self._next_id = 1

        self.z = z
        self.R = R
        self.T = T
        self.V = V
        self.focal = focal
        self.height = 1.75
        self.W = W
        self.H = H

    def predict(self):
        """
        Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()
            # track.predict_box(self.kf_box)

    def update(self, AllDetections):
        tracks_save = np.zeros(shape=(0, 6))
        if len(AllDetections) == 0:
            return tracks_save

        # confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        # unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        confirmed_tracks = [i for i, t in enumerate(self.tracks)]

        matches_a, unmatched_tracks_a, unmatched_detections = self._match(AllDetections, confirmed_tracks)
        # print('matches_a', matches_a)
        # print('unmatched_tracks_a', unmatched_tracks_a)
        # print('unmatched_detections_a', unmatched_detections_a)

        # position_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 2
        # ]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if self.tracks[k].time_since_update > 2
        # ]
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.second_position_match(self.tracks, AllDetections, position_track_candidates, unmatched_detections_a)

        # print('unmatched_tracks_b', unmatched_tracks_b)
        # matches = np.vstack((matches_a, matches_b))
        # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        matches = matches_a
        unmatched_tracks = unmatched_tracks_a
        # print('unmatched_tracks', unmatched_tracks, '\n')

        for m in matches:

            self.tracks[m[0]].update(AllDetections[m[1]])
            if self.tracks[m[0]].is_confirmed():
            # print('******************', self.tracks[m[0]].mean.shape)
                output_mean = self.tracks[m[0]].mean.reshape(self.tracks[m[0]].mean.shape[0])
                temp = [self.tracks[m[0]].track_id, output_mean[0], output_mean[1], output_mean[2], -1, -1]
                tracks_save = np.vstack((tracks_save, temp))

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # for track_idx in unmatched_tracks:
        #     # self.tracks[track_idx].mark_missed()
        #     # if self.tracks[track_idx].related_track is None:
        #     if self.tracks[track_idx].related_detection is None:
        #         self.tracks[track_idx].mark_missed()
        #     else:
        #         print('***************'*5)
        #         self.tracks[track_idx].collision_recover()

        for detection_idx in unmatched_detections:
            if AllDetections[detection_idx].fusion_state == 3 or AllDetections[detection_idx].fusion_state == 1:
                self._initiate_track(AllDetections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        return tracks_save

    def update_threeStage(self, AllDetections):
        tracks_save = np.zeros(shape=(0, 6))
        if len(AllDetections) == 0:
            return tracks_save

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        FusionDetections, CameraDetections, RadarDetections = [], [], []
        for detection in AllDetections:
            if detection.fusion_state == 3:
                FusionDetections.append(detection)
            elif detection.fusion_state == 1:
                CameraDetections.append(detection)
            elif detection.fusion_state == 2:
                RadarDetections.append(detection)

        matches_Fa, matches_Ca, matches_R, unmatched_tracks_a, unmatched_FusionDetections, unmatched_CameraDetections, unmatched_RadarDetections\
            = self.match_threeStage(FusionDetections, CameraDetections, RadarDetections, confirmed_tracks)

        # matches = matches_a
        unmatched_tracks = unmatched_tracks_a + unconfirmed_tracks

        matches_Fb, matches_Cb, unmatched_tracks, unmatched_FusionDetections, unmatched_CameraDetections = (
            linear_assignment.matching_init(self.tracks, FusionDetections, CameraDetections, unmatched_tracks,
                                            unmatched_FusionDetections, unmatched_CameraDetections))
        matches_F = np.concatenate((matches_Fa, matches_Fb), axis=0)
        matches_C = np.concatenate((matches_Ca, matches_Cb), axis=0)
        for m in matches_F:
            self.tracks[m[0]].time_since_recover = 0
            self.tracks[m[0]].update(FusionDetections[m[1]])
            if self.tracks[m[0]].is_confirmed():
            # print('******************', self.tracks[m[0]].mean.shape)
                output_mean = self.tracks[m[0]].mean.reshape(self.tracks[m[0]].mean.shape[0])
                temp = [self.tracks[m[0]].track_id, output_mean[0], output_mean[1], output_mean[2], -1, -1]
                tracks_save = np.vstack((tracks_save, temp))
        for m in matches_C:
            self.tracks[m[0]].time_since_recover = 0
            self.tracks[m[0]].update(CameraDetections[m[1]])
            if self.tracks[m[0]].is_confirmed():
            # print('******************', self.tracks[m[0]].mean.shape)
                output_mean = self.tracks[m[0]].mean.reshape(self.tracks[m[0]].mean.shape[0])
                temp = [self.tracks[m[0]].track_id, output_mean[0], output_mean[1], output_mean[2], -1, -1]
                tracks_save = np.vstack((tracks_save, temp))
        for m in matches_R:
            self.tracks[m[0]].time_since_recover = 0
            self.tracks[m[0]].update(RadarDetections[m[1]])
            if self.tracks[m[0]].is_confirmed():
            # print('******************', self.tracks[m[0]].mean.shape)
                output_mean = self.tracks[m[0]].mean.reshape(self.tracks[m[0]].mean.shape[0])
                temp = [self.tracks[m[0]].track_id, output_mean[0], output_mean[1], output_mean[2], -1, -1]
                tracks_save = np.vstack((tracks_save, temp))

        # for track_idx in unmatched_tracks:
        #     self.tracks[track_idx].mark_missed()


        for track_idx in unmatched_tracks:
            # self.tracks[track_idx].mark_missed()
            # if self.tracks[track_idx].related_track is None:
            if self.tracks[track_idx].time_since_recover <= 3:
                if self.tracks[track_idx].related_track is not None and getTrackDis(self.tracks[track_idx]) <= 4:
                    self.tracks[track_idx].collision_recover(reocver_state=1)
                elif self.tracks[track_idx].related_detection is not None:
                    self.tracks[track_idx].collision_recover(reocver_state=2)
                else:
                    self.tracks[track_idx].mark_missed()
            else:
                self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_FusionDetections:
            self._initiate_track(FusionDetections[detection_idx])
        for detection_idx in unmatched_CameraDetections:
            self._initiate_track(CameraDetections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        return tracks_save

    def collision_predict(self, extended_line_threshold=4, collision_threshold=3):

        # tracks_confirmed = []
        # for i in range(len(self.tracks)):
        #     if self.tracks[i].is_confirmed():
        #         tracks_confirmed.append(self.tracks[i])
        tracks_confirmed = [t for t in self.tracks if t.is_confirmed()]

        tracks_costmatrix = linear_assignment.get_tracks_costmatrix(tracks_confirmed)
        if tracks_costmatrix.shape[0] == 0:
            return

        for i in range(tracks_costmatrix.shape[0]):
            tracks_cost = tracks_costmatrix[i]
            distance_min = np.min(tracks_cost)
            index_min = np.argmin(tracks_cost)
            if distance_min < extended_line_threshold:
                # rel_det_x = (float(self.tracks[index_min].mean[0]) + float(self.tracks[i].mean[0])) / 2.0
                # rel_det_y = (float(self.tracks[index_min].mean[1]) + float(self.tracks[i].mean[1])) / 2.0
                rel_det_x = float(self.tracks[i].mean[0])
                rel_det_y = float(self.tracks[i].mean[1])
                self.tracks[i].related_detection = [rel_det_x, rel_det_y]
                # self.tracks[i].related_detection = [float(self.tracks[index_min].mean[0]), float(self.tracks[index_min].mean[1])]
                if self.tracks[i].velocity is None or self.tracks[index_min].velocity is None:
                    self.tracks[i].related_track = None
                    continue
                intersection_point = calculate_intersection_point(self.tracks[i], self.tracks[index_min])
                # # print('intersection_point', intersection_point, intersection_point.shape, type(intersection_point[0]))
                # # print('self.tracks[i].mean[0]', self.tracks[i].mean[0], type(self.tracks[i].mean[0]))
                distance_intersection = np.sqrt((self.tracks[i].mean[0] - intersection_point[0]) ** 2 + \
                                                (self.tracks[i].mean[1] - intersection_point[1]) ** 2)

                if distance_intersection < collision_threshold:
                    self.tracks[i].related_track = self.tracks[index_min]
                else:
                    self.tracks[i].related_track = None
            else:
                self.tracks[i].time_recover_frames += 1
                if self.tracks[i].time_recover_frames >= 3:
                    self.tracks[i].time_recover_frames = 0
                    self.tracks[i].related_track = None
                    self.tracks[i].related_detection = None



    def _match(self, AllDetections, confirmed_tracks):
        # Associate confirmed tracks using appearance features.
        velocities = np.array(
            [self.tracks[i].velocity if self.tracks[i].velocity is not None else np.array((0, 0)) for i in confirmed_tracks])  # (N_track, 2)  (norm_dy, norm_dx)
        k_observations = np.array(
            [k_previous_obs(self.tracks[i].observations, self.tracks[i].age, self.tracks[i].delta_t) for i in confirmed_tracks])  # (N_track, 3) (x, y, c)

        matches, unmatched_tracks, unmatched_detections\
            = linear_assignment.matching_cascade(self.tracks, AllDetections, velocities, k_observations, confirmed_tracks)

        return matches, unmatched_tracks, unmatched_detections


    def match_threeStage(self, FusionDetections, CameraDetections, RadarDetections, confirmed_tracks):
        # Associate confirmed tracks using appearance features.
        velocities = np.array(
            [self.tracks[i].velocity if self.tracks[i].velocity is not None else np.array((0, 0)) for i in confirmed_tracks])  # (N_track, 2)  (norm_dy, norm_dx)
        k_observations = np.array(
            [k_previous_obs(self.tracks[i].observations, self.tracks[i].age, self.tracks[i].delta_t) for i in confirmed_tracks])  # (N_track, 3) (x, y, c)

        matches_F, matches_C, matches_R, unmatched_tracks, unmatched_detections_Fusion, unmatched_detections_Camera, unmatched_detections_Radar \
            = linear_assignment.matching_cascade_threeStage(self.tracks, FusionDetections, CameraDetections,
                                                            RadarDetections, velocities, k_observations, confirmed_tracks)

        return matches_F, matches_C, matches_R, unmatched_tracks, unmatched_detections_Fusion, unmatched_detections_Camera, unmatched_detections_Radar

    def _initiate_track(self, detection):
        """
        create new tracker

        Parameters
        ----------
        detection : unmatched detection camera, radar has too much interference

        Returns
        -------

        """
        # detection_fusion_state = detection.fusion_state
        self.tracks.append(Track(measurement=detection))
