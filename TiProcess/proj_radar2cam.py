import numpy as np



def cam_to_radar(bboxs, radar2img_matrix, IntrinsicMatrix, radar_height=-1.0):
    """
    camera point project to radar coordinate
    :param bboxs : (N,4) , N is the number of bbox, tlwh
    :param z: estimated z, the height of the object
    :param R: rotate matrix, 3x3
    :param T: transmit matrix, 1,3
    :param V: camera intrinsic matrix, 3x3

    :return: [3xN],3 is (x,y,z)
    """
    V = np.array(IntrinsicMatrix)
    # RT_Matrix = np.array(np.dot(np.linalg.inv(IntrinsicMatrix), radar2img_matrix))
    RT_Matrix = np.array(radar2img_matrix)
    # print('RT_Matrix', RT_Matrix)
    R = RT_Matrix[0:3, 0:3]
    # T = RT_Matrix[:, 3:4].reshape((1, 3))
    T = np.array([RT_Matrix[0, 3], RT_Matrix[2, 3], RT_Matrix[1, 3]]).reshape((1, 3))

    # print('R', R)
    # print('T  ', T)
    # sy = np.sqrt(R[2, 0] ** 2 + R[2, 2] ** 2)
    # sy = np.sqrt(R[2, 0] ** 2 + R[2, 1] ** 2)
    # theta_y = np.arctan2(sy, R[2, 2])
    # theta_x = np.arctan2(R[1, 2], R[0, 2])
    # theta_z = np.arctan2(R[2, 1], -R[2, 0])

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    theta_y = np.arctan2(-R[2, 0], sy)
    theta_z = np.arctan2(R[1, 0], R[0, 0])
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    # print('theta_all: ', theta_x, theta_y, theta_z)
    u0 = V[0, 2]
    v0 = V[1, 2]
    fx = V[0, 0]
    fy = V[1, 1]
    cam_to_radar_point = []

    for i in range(len(bboxs)):
        u = bboxs[i, 0] + bboxs[i, 2] / 2
        v = bboxs[i, 1] + bboxs[i, 3]
        # theta_a = np.arctan((v - v0) / fy)
        theta_a = np.arctan((v - v0) / fy) + theta_y
        # theta_a = np.arctan((v - v0) / fy)
        test_y = radar_height / np.tan(theta_a)

        # theta_b = np.arctan((u - u0) / fx)
        theta_b = np.arctan((u - u0) / fx) + theta_x
        # theta_b = np.arctan((u - u0) / fx)
        test_x = test_y * np.tan(theta_b)
        cam_to_radar_point.append([test_x, test_y])
    # print('cam_to_radar_point', cam_to_radar_point)
    # for i in range(len(bboxs)):
    #     u = bboxs[i, 0] + bboxs[i, 2] / 2
    #     v = bboxs[i, 1] + bboxs[i, 3]
    #     cam_point = np.array([u, v, 1])
    #     cam_point = np.dot(np.linalg.inv(V), cam_point)
    #     cam_point = np.dot(np.linalg.inv(R), cam_point)
    #     # cam_point = np.dot(V.T, cam_point)
    #     # cam_point = np.dot(R.T, cam_point)
    #     cam_point = cam_point / cam_point[2]
    #     R_T = np.dot(np.linalg.inv(R), T)
    #     cam_to_radar_point = cam_point[:2] * (z + R_T[2]) - R_T[:2]
    # return np.array(cam_to_radar_point) + T[:, 0:2]
    # print('cam_to_radar_point', np.array(cam_to_radar_point))
    cam_to_radar_point = np.array(cam_to_radar_point)
    cam_to_radar_point = cam_to_radar_point - T[:, 0:2]
    return np.array(cam_to_radar_point)

# raw person_h:1.7
def cam_to_radar2(bboxs, radar2img_matrix, IntrinsicMatrix, person_h=1.9):
    """
        camera point project to radar coordinate
        :param bboxs : (N,4) , N is the number of bbox, tlwh
        :param z: estimated z, the height of the object
        :param R: rotate matrix, 3x3
        :param T: transmit matrix, 1,3
        :param V: camera intrinsic matrix, 3x3

        :return: [3xN],3 is (x,y,z)
        """

    # bboxs[:, 0] = bboxs[:, 0] + 225   # if crop
    bboxs[:, 0] = bboxs[:, 0]

    Internal_matrix = np.array(IntrinsicMatrix)
    # RT_Matrix = np.array(np.dot(np.linalg.inv(IntrinsicMatrix), radar2img_matrix[0:3])) # 3*4
    # # print('RT_Matrix', RT_Matrix)
    # R = RT_Matrix[:, 0:3]
    # T = RT_Matrix[:, 3:4].reshape((1, 3))

    u0 = Internal_matrix[0, 2]
    v0 = Internal_matrix[1, 2]
    fx = Internal_matrix[0, 0]
    fy = Internal_matrix[1, 1]

    xy_camera = []
    if bboxs.shape[0] > 0:
        dets_box = np.expand_dims(bboxs, 0)  # (1,m,4)
        u_detection = dets_box[..., 0] + dets_box[..., 2] / 2
        z_xy = person_h * fy / dets_box[..., 3]  # (1,m)
        x_xy = (u_detection - u0) * z_xy / fx
        xy_camera = np.concatenate((x_xy, z_xy))  # (2,m)
        # print("xy_camera:", xy_camera.shape)
        xy_TI = xy_camera.swapaxes(0, 1)  # shape: (m,2)
    # z_camera = np.ones((1, xy_camera.shape[1])) * person_h * -1
    # one_camera = np.ones((1, xy_camera.shape[1]))
    # xyz1_camera = np.concatenate((xy_camera, z_camera, one_camera))  # (4,m)
    # xyz_TI = np.dot(RT_Matrix, xyz1_camera)     # (3,m)
    # xy_TI = xyz_TI.swapaxes(0, 1)[:, 0:2]   # (m,2)

    return xy_TI
