import os
import argparse
import datetime
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from module.LidarBEVbackbone import LidarBEVBB



def HeatMapNMS(heat_map_scores, max_num_objects = 20, kernel_size = 3, score_threshold = 0.6):
    bs, cls_num, grid_height, grid_width = heat_map_scores.shape[0], heat_map_scores.shape[1], heat_map_scores.shape[2], heat_map_scores.shape[3]
    my_maxpool = nn.MaxPool2d(kernel_size=(kernel_size, kernel_size), stride=1, padding=1)
    # max_scores = nn.MaxPool2d(data, kernel_size=3, padding=0)
    max_scores = my_maxpool(heat_map_scores)
    peak_mask = torch.eq(heat_map_scores, max_scores).type(torch.float)
    filtered_heat_map = heat_map_scores * peak_mask

    flattened_t_heat_map = torch.reshape(filtered_heat_map, [1, 1, -1])
    _, batch_top_k_indices = torch.topk(flattened_t_heat_map, k=max_num_objects, sorted=True)
    batch_top_k_indices = batch_top_k_indices.transpose(1, 2)
    batch_index = torch.arange(1, dtype=torch.float) * grid_height * grid_width * 1
    batch_index = batch_index.cuda()
    tmp = torch.arange(1, dtype=torch.float).cuda()

    flattened_indices = (
            torch.reshape(batch_index, [bs, 1, 1]) + batch_top_k_indices * 1 +
            torch.reshape(tmp, [1, 1, 1]))
    flattened_indices = torch.reshape(flattened_indices, [-1]).type(torch.int)
    top_k_indices = torch.as_tensor(np.array(np.unravel_index(flattened_indices.cpu().detach(), (grid_height, grid_height))))
    # top_k_indices = py_utils.HasShape(
    # tf.transpose(top_k_indices), [bs * max_num_objects * nms_cls, 4])
    top_k_indices = torch.reshape(torch.transpose(top_k_indices, 0, 1), [bs * max_num_objects * 1, 2])
    # peak_heat_map = filtered_heat_map
    # filtered_heat_map.data = torch.where(filtered_heat_map.data >= score_threshold, filtered_heat_map.data, 1e-4)

    # print('top_k_indices:\n', top_k_indices)
    # print('peak_heat_map:\n', peak_heat_map)

    return top_k_indices


# heat_map_scores = [[1., 7., 5., 7., 5.],
#        [1.5, 9., 5.9, 8., 5.],
#        [7., 7.2, 7.3, 5., 5.],
#        [0.5, 0.6, 5., 7.4, 5.],
#        [0.74, 0.6, 5., 1., 5.]]
# heat_map_scores = np.array(heat_map_scores).reshape(1, 1, 5, 5)
# heat_map_scores = torch.from_numpy(heat_map_scores)
#
# kernel_size = 3
# max_num_objects = 6
# score_threshold = 5.
#
#
# top_k_indices, peak_heat_map = HeatMapNMS(heat_map_scores, kernel_size, max_num_objects, score_threshold)
#
# feature_map = [[[1., 7., 5., 7., 5.],
#        [1.5, 9., 5.9, 8., 5.],
#        [7., 7.2, 7.3, 5., 5.],
#        [0.5, 0.6, 5., 7.4, 5.],
#        [0.74, 0.6, 5., 1., 5.]],
#        [[-1., -7., -5., -7., -5.],
#        [-1.5, -9., -5.9, -8., -5.],
#        [-7., -7.2, -7.3, -5., -5.],
#        [-0.5, -0.6, -5., -7.4, -5.],
#        [-0.74, -0.7, -5., -1., -5.]]]
# feature_map = np.array(feature_map).reshape(1, 2, 5, 5)
# feature_map = torch.from_numpy(feature_map)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, width, height):
        # x = tensor_list.tensors
        not_mask = torch.ones((1, width, height))
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # print('position_embedding_shape:', pos.shape)
        return pos

# pos_feature_dim = 64
# PositionEmbedding = PositionEmbeddingSine(pos_feature_dim)
# AllPositionEmbedding = PositionEmbedding(top_k_indices)      # (1, 128, 128, 128) (bs, pe_dim, width, height)
#
#
# all_feature = torch.Tensor(1, 2+pos_feature_dim*2)
# for i in range(top_k_indices.shape[0]):
#     dim_x, dim_y = int(top_k_indices[i][0]), int(top_k_indices[i][1])
#     img_feature = feature_map[0, :, dim_x, dim_y].reshape(1, 2)
#     pe_xy = AllPositionEmbedding[0, :, dim_x, dim_y].reshape(1, pos_feature_dim*2)
#     img_feature = torch.cat((img_feature, pe_xy), 1)
#     all_feature = torch.cat((all_feature, img_feature), 0)
#
# all_feature = all_feature[1:all_feature.shape[0]]
# print('all_feature:\n', all_feature.shape)

class SingleBranchTransformer(nn.Module):

    def __init__(self, d_model, nhead=2, encoder_num_layers=2, decoder_num_layers=2):
        """Constructor for Single Branch TADN Transformer model

        Args:
            d_model (int): Number of expected features in the transformer inputs
            nhead (int, optional): Number of heads. Defaults to 2.
            encoder_num_layers (int, optional): Number of encoder layers. Defaults to 2.
            decoder_num_layers (int, optional): Number of decoder layers. Defaults to 2.
        """

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_num_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_num_layers
        )

        self.d_model = d_model

    def forward(
        self,
        targets,
        detections,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            targets (torch.Tensor): (num_K, batch_size, d_model) Targets input stream. Includes null-target.
            detections (torch.Tensor): (num_K, batch_size, d_model) Detections input stream.
            src_key_padding_mask (Optional[torch.Tensor], optional): Mask for src keys per batch. Defaults to None.
            tgt_key_padding_mask (Optional[torch.Tensor], optional): Mask for tgt keys per batch. Defaults to None.
            memory_mask (Optional[torch.Tensor], optional): Additive mask for the encoder output. Defaults to None.

        Returns:
            torch.Tensor: (num_targets+1, d_model) targets output stream.
            torch.Tensor: (num_detections, d_model) detections output stream.
        """
        transformed_targets = self.encoder(
            targets, src_key_padding_mask=src_key_padding_mask
        )
        transformed_detections = self.decoder(
            detections, transformed_targets, tgt_key_padding_mask=tgt_key_padding_mask
        )

        return transformed_detections

print(' ')
