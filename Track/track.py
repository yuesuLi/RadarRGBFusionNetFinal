# vim: expandtab:ts=4:sw=4
# -*- coding: utf-8 -*-
import numpy as np
from .kalman_filter_fusion import KalmanFilter_fusion
from TiProcess.proj_radar2cam import cam_to_radar2

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    NeedRecover = 4


class FusionState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    only_camera = 1
    only_radar = 2
    fusion_sensors = 3


def speed_direction(pre_xy, curr_xy):
    x1, y1 = pre_xy[0], pre_xy[1]
    x2, y2 = curr_xy[0], curr_xy[1]
    speed = np.array([y2-y1, x2-x1])
    norm = np.sqrt((y2-y1)**2 + (x2-x1)**2) + 1e-6
    # (2,)  (norm_dy, norm_dx)
    return speed / norm



class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    (x, y, a, h, vx, vy, va,vh)8维变量

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
        初始状态分布的mean vector
    covariance : ndarray
        Covariance matrix of the initial state distribution.初始状态分布的协方差矩阵
    track_id : int
        A unique track identifier.唯一id认证
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
        必须要连续检测到n_init帧, 目标才会转入confirmed状态, 否则就是deleted状态
        在confirmed之前的连续探测次数。如果在n_init的第一帧就消失，放入deleted
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
        在轨迹被设置为Deleted前的最大连续miss次数
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
        这个轨迹起源的检测的特征矢量。If not none, 特征会被加入'feature' cache


    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.测量更新总数
    age : int
        Total number of frames since first occurance.总帧数
    time_since_update : int
        Total number of frames since last measurement update.最后一个测量更新后的总帧数
    state : TrackState
        The current track state.当前轨迹状态
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
        每次测量更新，关联的feature vector被加入list

    """

    count = 0

    def __init__(self, measurement, n_init=3, max_age=3, delta_t=3, pointstate=0,
                 z=None, R=None, T=None, V=None, mean_box=None, covariance_box=None):

        # self.mean = mean
        # self.covariance = covariance
        self.kf = KalmanFilter_fusion(dim_x=6, dim_z=3)
        self.mean, self.covariance = self.kf.initiate(measurement=measurement)
        # self.target_state = self.mean
        # self.pointstate = pointstate

        self.track_fusion_state = measurement.fusion_state

        self.state = TrackState.Tentative
        self.track_id = Track.count
        Track.count += 1
        self.age = 1
        self.time_since_update = 0
        self.time_since_img_update = 0
        self.time_since_recover = 0
        self.time_recover_frames = 0
        self.assigned = 0       # 由于是多阶段匹配，用该参数表示track是否已与检测匹配，0代表未匹配，1代表已匹配
        self.history = []
        self.hits = 1
        self.hit_streak = 0

        # self.pre_xy = None
        self.velocity = None
        self.track_orientation = 0
        """
            placeholder, (x, y, confidence), confidence == -1 means non-observation status
        """
        self.last_observation = np.array([-100, -100, -1])
        self.observations = dict()
        self.history_observations = []
        self.delta_t = delta_t

        self._n_init = n_init
        self._max_age = max_age

        self.tlwh = measurement.tlwh
        self.proj_xy = measurement.proj_xy
        self.classes = measurement.classes
        self.feature = measurement.feature
        self.confidence = measurement.confidence
        self.r_center = measurement.r_center
        self.r_feature = measurement.r_feature

        self.height = 1.75
        self.related_track = None
        self.related_detection = None
        # self.z = z
        # self.R = R
        # self.T = T
        # self.V = V

        self.mean_box = mean_box
        self.covariance_box = covariance_box

    def predict(self):
        """
        Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = self.kf.predict()

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self.kf.mean)

    def predict_box(self, kf_box):
        self.mean_box, self.covariance_box = kf_box.predict(self.mean_box, self.covariance_box)

    def update(self, match_detection, associate_state=None):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # # if only cam, state = 1; if only radar, state = 2; if cam + radar, state = 3
        # if detection_fusion.fusion_state == 2:    # radar
        #     if self.pointstate == 1:
        #         self.pointstate = 3
        #     position_state = 'radar'
        # if detection_fusion.fusion_state == 1:    # cam
        #     f = self.feature
        #     b = self.tlwh
        #     # pointstate==2 means current time only radar without camera
        #     if self.pointstate == 2:
        #         self.feature = detection_fusion.feature
        #         self.tlwh = detection_fusion.tlwh
        #     else:
        #         self.feature = detection_fusion.feature * 0.5 + f * 0.5
        #         self.tlwh = (detection_fusion.tlwh + b) / 2
        #     self.classes = detection_fusion.classes
        #     self.confidence = detection_fusion.confidence
        #     if self.pointstate == 2:
        #         self.pointstate = 3
        #     position_state = 'cam'
        # if detection_fusion.fusion_state == 3:
        #     f = self.feature
        #     b = self.tlwh
        #     if self.pointstate == 2:
        #         self.feature = detection_fusion.feature
        #     else:
        #         # self.feature = detection.feature * 0.5 + f * 0.5
        #         # self.box = (detection.box + b) / 2
        #         self.feature = detection_fusion.feature
        #         self.tlwh = detection_fusion.tlwh
        #     self.classes = detection_fusion.classes
        #     self.confidence = detection_fusion.confidence
        #     self.tlwh = detection_fusion.tlwh
        #     self.pointstate = 3
        #
        #     position_state = 'radar'

        self.track_fusion_state = match_detection.fusion_state

        if match_detection.fusion_state == 3:           # camera + radar
            self.time_since_img_update = 0
            self.tlwh = match_detection.tlwh
            self.proj_xy =match_detection.proj_xy
            self.classes = match_detection.classes
            self.feature = match_detection.feature
            self.confidence = match_detection.confidence
            self.r_center = match_detection.r_center
            self.r_feature = match_detection.r_feature
        elif match_detection.fusion_state == 1:         # only camera
            self.time_since_img_update = 0
            self.tlwh = match_detection.tlwh
            self.proj_xy = match_detection.proj_xy
            self.classes = match_detection.classes
            self.feature = match_detection.feature
            self.confidence = match_detection.confidence
        elif match_detection.fusion_state == 2:         # only radar
            self.time_since_img_update += 1
            self.r_center = match_detection.r_center
            self.r_feature = match_detection.r_feature

        if self.time_since_img_update >= self._max_age:
            self.state = TrackState.Tentative
            return

        # curr_target_xy = match_detection.center[0:2]
        curr_target_xyc = np.array([float(match_detection.center[0]), float(match_detection.center[1]), 1])
        if self.last_observation.sum() >= -190:
            pre_xyc = None
            # 从self.age - self.delta_t 年龄开始找检测，默认是self.age-3到self.age-1
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt in self.observations:
                    pre_xyc = self.observations[self.age - dt]
                    break
            if pre_xyc is None:
                pre_xyc = self.last_observation
            self.velocity = speed_direction(pre_xyc, curr_target_xyc) # (2,)  (norm_dy, norm_dx)
            self.track_orientation = np.arctan2(self.velocity[0], self.velocity[1])
            # self.track_orientation = np.arctan(self.velocity[0], self.velocity[1])

            """
              Insert new observations.
            """
        self.last_observation = curr_target_xyc
        self.observations[self.age] = curr_target_xyc
        self.history_observations.append(curr_target_xyc)

        self.time_since_update = 0

        # self.time_since_recover = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        measure = np.array([float(match_detection.center[0]), float(match_detection.center[1]), float(self.track_orientation)])
        self.mean, self.covariance = self.kf.update(measure)

        # self.pre_xy = np.array([xy[0], xy[1]])
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # self.kf.update(match_detection, self.velocity)

    def collision_recover(self, reocver_state):
        self.time_since_recover += 1
        if reocver_state == 1:
            self.track_fusion_state = self.related_track.track_fusion_state
            measure = np.array([float(self.related_track.mean[0]), float(self.related_track.mean[1]), float(self.track_orientation)])
        elif reocver_state == 2:
            # measure = np.array([float(recover_track.mean[0]), float(recover_track.mean[1]), float(self.track_orientation)])
            measure = np.array([self.related_detection[0], self.related_detection[1], float(self.track_orientation)])

        # curr_target_xyc = np.array([float(measure[0]), float(measure[1]), 1])
        # if self.last_observation.sum() >= -190:
        #     pre_xyc = None
        #     # 从self.age - self.delta_t 年龄开始找检测，默认是self.age-3到self.age-1
        #     for i in range(self.delta_t):
        #         dt = self.delta_t - i
        #         if self.age - dt in self.observations:
        #             pre_xyc = self.observations[self.age - dt]
        #             break
        #     if pre_xyc is None:
        #         pre_xyc = self.last_observation
        #     self.velocity = speed_direction(pre_xyc, curr_target_xyc)  # (2,)  (norm_dy, norm_dx)
        #     self.track_orientation = np.arctan2(self.velocity[0], self.velocity[1])
        # self.last_observation = curr_target_xyc
        # self.observations[self.age] = curr_target_xyc
        # self.history_observations.append(curr_target_xyc)

        self.mean, self.covariance = self.kf.update(measure)

        # self.pre_xy = np.array([xy[0], xy[1]])
        # if self.state == TrackState.Tentative and self.hits >= self._n_init:
        #     self.state = TrackState.Confirmed
        # if self.time_since_recover >= self._max_age:
        # self.state = TrackState.Tentative

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
