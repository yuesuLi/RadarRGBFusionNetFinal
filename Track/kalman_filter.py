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


class KalmanFilter(object):

    def __init__(self):
        dt = 0.1
        self.dt = 0.1
        self._motion_mat = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        self._std_weight_position = 1. / 20     # 位置的方差
        self._std_weight_velocity = 1. / 160    # 速度的方差

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.
        """
        mean = np.array([measurement.center[0], measurement.center[1], 0, 0])
        # std = [0.1, 0.1, 0.1, 0.1]
        std = [self._std_weight_position, self._std_weight_position, self._std_weight_velocity, self._std_weight_velocity]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
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
        motion_cov = np.diag(np.square([0.005, 0.005, 0.005, 0.005]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, position_state):
        """Project state distribution to measurement space.
        将状态分布从状态空间投射到观测空间

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
            根据状态估计，返回投影后的均值（4x1）和协方差(4x4)矩阵

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

    def update(self, mean, covariance, measurement, position_state):
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
        projected_mean, projected_cov = self.project(mean, covariance, position_state)
        # predict
        # 2x1 4x4

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)

        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # kalman_gain = np.linalg.multi_dot((covariance, self._update_mat.T, scipy.linalg.inv(projected_cov)))
        # innovation_cov = np.diag(np.square([0.1, 0.1, 0.1]))
        r = np.linalg.norm(measurement[:2])
        phi = math.atan2(measurement[1], measurement[0])
        # v = measurement[2]*np.cos(phi)+measurement[3]*np.sin(phi)
        v = measurement[2]
        mea = np.array([r, phi, v])
        innovation = mea - projected_mean
        # mea is observation

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # new_covariance = covariance - np.linalg.multi_dot((
        #     kalman_gain, projected_cov, kalman_gain.T))
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, self._update_mat, covariance))
        return new_mean, new_covariance

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
        mean, covariance = self.project(mean, covariance)
        # 投影到观测空间
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        mean, covariance = mean[:2], covariance[:2, :2]
        measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
