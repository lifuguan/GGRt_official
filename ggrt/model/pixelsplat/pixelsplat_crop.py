from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn, optim

from .types import Gaussians
from ...dataset.data_module import get_data_shim
from ...dataset.types import BatchedExample
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from math import ceil
import copy

@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass

def random_crop(data,size=[160,224] ,center=None):
    _,_,_,h, w = data['context']['image'].shape
    # size=torch.from_numpy(size)
    batch=copy.deepcopy(data)
    out_h, out_w = size[0], size[1]

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)
    batch['context']['image'] = batch['context']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]
    # batch['target']['image'] = batch['target']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]

    batch['context']['intrinsics'][:,:,0,0]=batch['context']['intrinsics'][:,:,0,0]*w/out_w
    batch['context']['intrinsics'][:,:,1,1]=batch['context']['intrinsics'][:,:,1,1]*h/out_h
    batch['context']['intrinsics'][:,:,0,2]=(batch['context']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    batch['context']['intrinsics'][:,:,1,2]=(batch['context']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h

    # batch['target']['intrinsics'][:,:,0,0]=batch['target']['intrinsics'][:,:,0,0]*w/out_w
    # batch['target']['intrinsics'][:,:,1,1]=batch['target']['intrinsics'][:,:,1,1]*h/out_h
    # batch['target']['intrinsics'][:,:,0,2]=(batch['target']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    # batch['target']['intrinsics'][:,:,1,2]=(batch['target']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h



    return batch

class PixelSplat(nn.Module):
    encoder: nn.Module
    decoder: Decoder
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        encoder_visualizer: Optional[EncoderVisualizer],
    ) -> None:
        super().__init__()

        # Set up the model.
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_visualizer = encoder_visualizer
        
        self.data_shim = get_data_shim(self.encoder)

    def forward(self, batch, global_step: int):
        # batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        crop_h=176
        crop_w=240
        row=ceil(h/crop_h)
        col=ceil(w/crop_w)
        # gt_full=data['target']['image'].squeeze(0).squeeze(0)
        for i in range(row):
            for j in range(col):
                if i==row-1 and j==col-1:
                    data_crop=random_crop(  batch,size=[crop_h,crop_w],center=(int(h-crop_h//2),int(w-crop_w//2)))
                elif i==row-1:#最后一行
                    data_crop=random_crop(  batch,size=[crop_h,crop_w],center=(int(h-crop_h//2),int(crop_w//2+j*crop_w)))
                elif j==col-1:#z最后一列
                    data_crop=random_crop( batch,size=[crop_h,crop_w],center=(int(crop_h//2+i*crop_h),int(w-crop_w//2)))
                else:
                    data_crop=random_crop( batch,size=[crop_h,crop_w],center=(int(crop_h//2+i*crop_h),int(crop_w//2+j*crop_w)))  
                # Run the model.
                for k in range(batch["context"]["image"].shape[1] - 1):
                    tmp_batch = self.batch_cut(data_crop["context"],k)
                    tmp_gaussians = self.encoder(tmp_batch, global_step, False)
                    if k == 0 and i+j==0:
                        gaussians: Gaussians = tmp_gaussians
                    else:
                        gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
                        gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
                        gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
                        gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
                    
            
        # gaussians = self.encoder(batch['context'], global_step, False)
            
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode='depth'
        )
        
        ret = {'rgb': output.color, 'depth': output.depth}
        target_gt = {'rgb': batch["target"]["image"]}
        return ret, target_gt
    
    def batch_cut(self, batch, i):
        return {
            'extrinsics': batch['extrinsics'][:,i:i+2,:,:],
            'intrinsics': batch['intrinsics'][:,i:i+2,:,:],
            'image': batch['image'][:,i:i+2,:,:,:],
            'near': batch['near'][:,i:i+2],
            'far': batch['far'][:,i:i+2],
            'index': batch['index'][:,i:i+2],
        }