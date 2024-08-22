import os
import argparse
import datetime
import shutil

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from module.LidarBEVbackbone import LidarBEVBB
from module.TRL import HeatMapNMS, PositionEmbeddingSine, SingleBranchTransformer


class LidarBEVDetection(nn.Module):
    
    def __init__(self):
        super(LidarBEVDetection, self).__init__()

        self.backbone = LidarBEVBB()
        self.PositionEncoder = PositionEmbeddingSine()
        self.TRL = SingleBranchTransformer(d_model=256)

        self.head_hm = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_wh = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        self.head_ori = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )


    def forward(self, LidarBEV, pre_topk_feature = None):

        feature_map, one_stage_hm = self.backbone(LidarBEV)   # (1, 128, 128, 128)  (bs, feature_dim, width, height)
        AllPositionEmbedding = self.PositionEncoder(feature_map.shape[2], feature_map.shape[3]).cuda()  # (1, 128, 128, 128)  (bs, pe_dim, width, height)
        top_k_indices = HeatMapNMS(one_stage_hm)

        pe_dim = AllPositionEmbedding.shape[1]
        fm_dim = feature_map.shape[1]
        all_feature = torch.Tensor(1, pe_dim + fm_dim).cuda()
        for i in range(top_k_indices.shape[0]):
            dim_x, dim_y = int(top_k_indices[i][0]), int(top_k_indices[i][1])
            img_feature = feature_map[0, :, dim_x, dim_y].reshape(1, fm_dim)
            pe_xy = AllPositionEmbedding[0, :, dim_x, dim_y].reshape(1, pe_dim)
            img_feature = torch.cat((img_feature, pe_xy), 1)
            all_feature = torch.cat((all_feature, img_feature), 0)

        all_feature = all_feature[1:all_feature.shape[0]]
        # print('all_feature:\n', all_feature.shape)  # (topk_num, 256)

        if pre_topk_feature is None:
            return all_feature

        all_feature = all_feature.reshape(all_feature.shape[0], 1, all_feature.shape[1])
        pre_topk_feature = pre_topk_feature.reshape(pre_topk_feature.shape[0], 1, pre_topk_feature.shape[1])
        final_topk_feature = self.TRL(all_feature, pre_topk_feature)    # (topk_num, 1, 256)

        for i in range(top_k_indices.shape[0]):
            dim_x, dim_y = int(top_k_indices[i][0]), int(top_k_indices[i][1])
            feature_map[0, :, dim_x, dim_y].data = final_topk_feature[i, 0, 0:128].data

        final_topk_feature = final_topk_feature.reshape(all_feature.shape[0], all_feature.shape[2])# (topk_num, 256)
        # hm_feature = self.head_hm(feature_map)  # (B, 1, 128, 128)
        wh_map = self.head_wh(feature_map)    # (B, 2, 128, 128)
        ori_map = self.head_ori(feature_map)  # (B, 2, 128, 128)

        return final_topk_feature, one_stage_hm, wh_map, ori_map
