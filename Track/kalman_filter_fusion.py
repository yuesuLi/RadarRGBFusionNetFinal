# vim: expandtab:ts=4:sw=4
#-*- coding: utf-8 -*-
import numpy as np
import scipy.linalg
import math


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
具有N个自由度的卡方分布的0.95分位数的表（包含N = 1，...，9的值）。取自MATLAB / Octave的chi2inv函数，用作Mahalanobis门控阈值。
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter_fusion(object):
    """
    dim_x: targrt state dim (x,y,o,vx,vy,vo)
    dim_z: detection dim (x,y,o)

    """

    def __init__(self, dim_x=6, dim_z=3):
        # dt = 0.5
        self.dt = 1
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.mean = np.zeros((dim_x, 1))  # state
        self.covariance = np.eye(dim_x)  # uncertainty covariance

        # F
        self._motion_mat = np.array([[1, 0, 0, self.dt, 0, 0],
                                     [0, 1, 0, 0, self.dt, 0],
                                     [0, 0, 1, 0, 0, self.dt],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])

        self._std_weight_position = 1. / 20         # 位置的方差
        self._std_weight_position_velocity = 1.     # 位置速度的方差
        self._std_weight_orientation_velocity = 1. / 50    # 方向速度的方差
        self._std_weight_orientation = 1. / 20     # 方向的方差s

        self._I = np.eye(dim_x)     # identity matrix. Do not alter this.
        self.Q = np.diag(np.square([0.005, 0.005, 0.05, 0.005, 0.005, 0.05]))     # process uncertainty
        self.B = None  # control transition matrix
        self.F = np.eye(self.dim_x)  # state transition matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])  # measurement function, np.array(dim_z, dim_x)
        self.R = np.diag(np.square([0.005, 0.005, 0.05]))  # measurement uncertainty
        self._alpha_sq = 1.  # fading memory control
        self.M = np.zeros((dim_x, self.dim_z))  # process-measurement cross correlation
        self.z = np.array([[None] * self.dim_z]).T
        self.K = np.zeros((self.dim_x, self.dim_z))  # 卡尔曼Kalman gain of the update step, np.array(dim_x, dim_z)
        self.S = np.zeros((self.dim_z, self.dim_z))  # 系统不确定度(P投影到测量空间),只读
        self.SI = np.zeros((self.dim_z, self.dim_z))  # inverse system uncertainty


    def initiate(self, measurement):
        """
        Create track from unassociated measurement.
        """
        self.mean = np.array([measurement.center[0], measurement.center[1], 0, 0, 0, 0]).reshape(self.dim_x, 1)
        # std = [0.1, 0.1, 0.1, 0.1]
        std = [self._std_weight_position, self._std_weight_position, self._std_weight_orientation,
               self._std_weight_position_velocity, self._std_weight_position_velocity,
               self._std_weight_orientation_velocity]
        self.covariance = np.diag(np.square(std))
        return self.mean, self.covariance

    def predict(self):
        """
        predict state

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step. 上一步中获得的8维mean vector
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step. 上一步获得的8x8协方差矩阵

        Returns
        -------
        mean, covariance : (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """

        self.mean = np.dot(self._motion_mat, self.mean)

        self.covariance = np.linalg.multi_dot((
            self._motion_mat, self.covariance, self._motion_mat.T)) + self.Q

        return self.mean, self.covariance


    def project(self, mean, covariance, position_state):
        """Project state distribution to measurement space.
        EKF: 将状态分布从状态空间投射到观测空间

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        """
        if position_state == 'radar':
            innovation_cov = np.diag(np.square([0.005, 0.005, 0.005]))
        elif position_state == 'cam':
            # innovation_cov = np.diag(np.square([0.05, 0.05, 0.05]))
            innovation_cov = np.diag(np.square([0.01, 0.01, 0.01]))

        # innovation_cov = np.diag(np.square([0.05, 0.05, 0.05]))
        r = np.linalg.norm(mean[:2])
        phi = math.atan2(mean[1], mean[0])
        v = mean[2]*np.cos(phi)+mean[3]*np.sin(phi)
        self._update_mat = np.array([[np.cos(phi), np.sin(phi), 0, 0],
                                     [-np.sin(phi)/r, np.cos(phi)/r, 0, 0],
                                     [np.cos(phi)*(mean[2]*mean[1]-mean[0]*mean[3])/r**2, np.sin(phi)*(mean[0]*mean[3]-mean[1]*mean[2])/r**2, np.cos(phi), np.sin(phi)]])
        # self._update_mat = np.array([[np.cos(phi), np.sin(phi), 0, 0],
        #                              [-np.sin(phi)/r, np.cos(phi)/r, 0, 0]])

        # new_mean = np.dot(self._update_mat, mean)
        new_mean = np.array([r, phi, v])
        # 2x4  dot 4x1
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return new_mean, covariance + innovation_cov

    def update(self, measurement, position_state=None):
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).状态协方差（8x8）
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box. 测量（4维）

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        PHT = np.dot(self.covariance, self.H.T)     # (dim_x, dim_z)
        # project system uncertainty into measurement space
        self.S = np.dot(self.H, PHT) + self.R   # (dim_z, dim_z)
        self.SI = np.linalg.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)   # (dim_x, dim_z)
        # detection_state is observation
        # detection_position = np.array([measurement.center[0:2]])
        # detection_orientation =
        # print('measurement', type(measurement[0:3]), measurement[0:3].shape)
        detection_state = measurement[0:3].reshape(self.dim_z, 1)

        innovation = detection_state - np.dot(self.H, self.mean)    # (dim_z, 1)
        self.mean = self.mean + np.dot(self.K, innovation)
        I_KH = self._I - np.dot(self.K, self.H)
        # self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        self.covariance = np.dot(I_KH, self.covariance)

        # save measurement and posterior state
        # self.z = deepcopy(z)
        # self.x_post = self.x.copy()
        # self.P_post = self.P.copy()
        return self.mean, self.covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        计算状态分布和测量之间的门限值，一个合适的距离门限值可以从ch2inv95获得。
        如果‘only_position’是false, 卡方分布有4个自由度，否则是2个


        Parameters
        ----------
        mean : ndarray
            基于状态分布的mean vector (8 dimensional).——预测值
        covariance : ndarray
            状态分布的协方差矩阵 (8x8 dimensional).——预测值
        measurements : ndarray
            Nx4矩阵，第二维4个向量为（x, y, a, h）——观测值
        only_position : Optional[bool]
            If True， 距离计算只考虑bounding box的中心位置

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
            返回一个长度为N的向量，第i个元素包含了（mean，covariance）和measurements[i]之间的马氏距离

        """
        # mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = self.mean[:2], self.covariance[:2, :2]
            measurements = measurements[:, :2]
        else:
            mean, covariance = self.mean[:3], self.covariance[:3, :3]
            measurements = measurements[:, :3]

        d = measurements - mean
        # z = scipy.linalg.solve_triangular(
        #     cholesky_factor, d.T, lower=True, check_finite=False,
        #     overwrite_b=True)
        # squared_maha = np.sum(z * z, axis=0)
        return d
