# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

from utils_loc import img2mse
from ggrt.geometry.depth import depth2inv


class MaskedL2ImageLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        if 'mask' in outputs:
            pred_mask = outputs['mask'].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss


def pseudo_huber_loss(residual, scale=10):
    trunc_residual = residual / scale
    return torch.sqrt(trunc_residual * trunc_residual + 1) - 1


class FeatureMetricLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, target_rgb_feat, nearby_view_rgb_feat, mask=None):
        '''
        Args:
            target_rgb_feat: [n_rays, n_samples=1, n_views+1, d+3]
            nearby_view_rgb_feat: [n_rays, n_samples=1, n_views+1, d+3]
        '''
        if mask is None:
            l1_loss = nn.L1Loss(reduction='mean')
            # mse_loss = nn.MSELoss(reduction='mean')
            # loss = mse_loss(nearby_view_rgb_feat, target_rgb_feat)

            loss = l1_loss(nearby_view_rgb_feat, target_rgb_feat)
        
        else:
            feat_diff = target_rgb_feat - nearby_view_rgb_feat
            feat_diff_square = (feat_diff * feat_diff).squeeze(1)
            mask = mask.repeat(1, 1, 1).permute(2, 0, 1)
            n_views, n_dims = target_rgb_feat.shape[-2], target_rgb_feat.shape[-1]
            loss = torch.sum(feat_diff_square * mask) / (torch.sum(mask.squeeze(-1)) * n_views * n_dims + 1e-6)

            # feat_diff_huber = pseudo_huber_loss(feat_diff, scale=0.8).squeeze(1)
            # mask = mask.repeat(1, 1, 1).permute(2, 0, 1)
            # n_views, n_dims = target_rgb_feat.shape[-2], target_rgb_feat.shape[-1]
            # loss = torch.sum(feat_diff_huber * mask) / (torch.sum(mask.squeeze(-1)) * n_views * n_dims + 1e-6)
        
        return loss


def self_sup_depth_loss(inv_depth_prior, rendered_depth, min_depth, max_depth):
    min_disparity = 1.0 / max_depth
    max_disparity = 1.0 / min_depth
    valid = ((inv_depth_prior > min_disparity) & (inv_depth_prior < max_disparity)).detach()

    inv_rendered_depth = depth2inv(rendered_depth)

    loss_depth = torch.mean(valid * torch.abs(inv_depth_prior - inv_rendered_depth))

    return loss_depth


def sup_depth_loss(ego_motion_inv_depths, gt_depth, min_depth, max_depth):
    num_iters = len(ego_motion_inv_depths)
    total_loss = 0
    total_w = 0
    gamma = 0.85
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth

    gt_inv_depth = depth2inv(gt_depth)

    valid = ((gt_inv_depth > min_disp) & (gt_inv_depth < max_disp)).detach()

    for i, inv_depth in enumerate(ego_motion_inv_depths):
        w = gamma ** (num_iters - i - 1)
        total_w += w

        loss_depth = torch.mean(valid * torch.abs(gt_inv_depth - inv_depth.squeeze(0)))
        loss_i = loss_depth
        total_loss += w * loss_i
    loss = total_loss / total_w
    return loss
