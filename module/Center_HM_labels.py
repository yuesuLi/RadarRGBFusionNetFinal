# draw heatmap for every GT
import numpy as np
import json



def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


# generate a 2D gaussian heatmap with a specified radius
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)


    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # avoid out of bounds
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]    # raw hm need draw region
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]  # gaussian hm region
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap) # output replace masked_heatmap

    return heatmap

def get_OneStage_lidar_annotation(lidar_json_path):

    with open(lidar_json_path) as f:
        lidar_json_data = json.load(f)
    x, y, z, l, w, h, alpha = [], [], [], [], [], [], []
    for idx in range(len(lidar_json_data['annotation'])):
        if lidar_json_data['annotation'][idx]['x'] >= 50 or lidar_json_data['annotation'][idx]['x'] <= 0\
                or lidar_json_data['annotation'][idx]['y'] >= 25 or lidar_json_data['annotation'][idx]['y'] <= -25:
            continue
        x.append(lidar_json_data['annotation'][idx]['x'])
        y.append(lidar_json_data['annotation'][idx]['y'])
        l.append(lidar_json_data['annotation'][idx]['l'])
        w.append(lidar_json_data['annotation'][idx]['w'])
        alpha.append(lidar_json_data['annotation'][idx]['alpha'])

    width = 128
    height = 128

    x = [tmp_x / 50 * width for tmp_x in x]
    y = [(-tmp_y + 25) / 50 * height for tmp_y in y]
    l = [tmp_l / 50 * width for tmp_l in l]
    w = [tmp_w / 50 * height for tmp_w in w]

    return x, y, l, w, alpha


def get_HM_label(x, y, l, w, alpha):
    hm = np.zeros((1, 128, 128), dtype=np.float32)

    for v, u, tmp_l, tmp_w, yaw in zip(x, y, l, w, alpha):
        min_radius = 2
        tmp_h = tmp_l * abs(np.cos(np.pi - yaw)) + tmp_w * abs(np.sin(np.pi - yaw))
        tmp_w = tmp_l * abs(np.sin(np.pi - yaw)) + tmp_w * abs(np.cos(np.pi - yaw))
        # print('tmp_h, tmp_w:', tmp_h, tmp_w)
        if tmp_h > 0 and tmp_w > 0:
            # print('num:', num)
            # num += 1
            radius = gaussian_radius((tmp_h, tmp_w), min_overlap=0.7) * 7   # too small
            # radius = radius * 15
            radius = max(min_radius, int(radius))

            # coor_x, coor_y = center[0] / down_sample_factor, center[1] / down_sample_factor
            ct = np.array([u, v], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[0], center=ct_int, radius=radius)

    return hm


def get_LW_label(x, y, l, w, alpha):
    lw_map = np.zeros((2, 128, 128), dtype=np.float32)

    for u, v, tmp_l, tmp_w, yaw in zip(x, y, l, w, alpha):
        if tmp_l > 0 and tmp_w > 0:

            # coor_x, coor_y = center[0] / down_sample_factor, center[1] / down_sample_factor
            ct = np.array([u, v], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            lw_map[0, ct_int[0], ct_int[1]] = tmp_l
            lw_map[1, ct_int[0], ct_int[1]] = tmp_w

    return lw_map

def get_Alpha_label(x, y, l, w, alpha):
    alpha_map = np.zeros((2, 128, 128), dtype=np.float32)

    for u, v, tmp_l, tmp_w, yaw in zip(x, y, l, w, alpha):
        if tmp_l > 0 and tmp_w > 0:
            # coor_x, coor_y = center[0] / down_sample_factor, center[1] / down_sample_factor
            ct = np.array([u, v], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            alpha_map[0, ct_int[0], ct_int[1]] = np.sin(yaw)
            alpha_map[1, ct_int[0], ct_int[1]] = np.cos(yaw)

    return alpha_map


def get_offset_label(x, y, l, w, alpha):
    offset_map = np.zeros((2, 128, 128), dtype=np.float32)

    for u, v, tmp_l, tmp_w, yaw in zip(x, y, l, w, alpha):
        if l > 0 and w > 0:
            # coor_x, coor_y = center[0] / down_sample_factor, center[1] / down_sample_factor
            ct = np.array([u, v], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            offset_map[0, ct_int[0], ct_int[1]] = np.sin(yaw)
            offset_map[1, ct_int[0], ct_int[1]] = np.cos(yaw)

    return offset_map