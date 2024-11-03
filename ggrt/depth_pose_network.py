import torch
import torch.nn as nn
import torch.nn.functional as F

from ggrt.geometry.camera import Camera
from ggrt.geometry.depth import inv2depth, disp_to_depth
from ggrt.optimizer import BasicUpdateBlockPose, BasicUpdateBlockDepth, DepthHead, PoseHead, UpMaskNet
from ggrt.pose_util import Pose
from ggrt.model.feature_network import ResNetEncoder
from ggrt.base.functools import partial


class DepthPoseNet(nn.Module):
    def __init__(self, pretrained=True, iters=12):
        super().__init__()

        self.iters = iters
        self.is_high = True
        self.out_normalize = True
        
        # get seq len in one stage. default: 4.
        self.seq_len = 4 
        
        # update iters
        self.iters = self.iters // self.seq_len
        
        # feature network, context network, and update block
        self.foutput_dim = 128
        self.feat_ratio = 8
        self.fnet = ResNetEncoder(out_chs=self.foutput_dim, stride=self.feat_ratio, pretrained=pretrained)
    
        self.depth_head = DepthHead(input_dim=self.foutput_dim, hidden_dim=self.foutput_dim, scale=False)
        self.pose_head = PoseHead(input_dim=self.foutput_dim * 2, hidden_dim=self.foutput_dim)
        self.upmask_net = UpMaskNet(hidden_dim=self.foutput_dim, ratio=self.feat_ratio)
        
        self.hdim = 128 if self.is_high else 64
        self.cdim = 32
        
        self.update_block_depth = BasicUpdateBlockDepth(
            hidden_dim=self.hdim,
            cost_dim=self.foutput_dim,
            ratio=self.feat_ratio,
            context_dim=self.cdim
        )
        self.update_block_pose = BasicUpdateBlockPose(hidden_dim=self.hdim, cost_dim=self.foutput_dim, context_dim=self.cdim)

        self.cnet_depth = ResNetEncoder(out_chs=self.hdim+self.cdim, stride=self.feat_ratio, pretrained=pretrained, num_input_images=1)
        self.cnet_pose = ResNetEncoder(out_chs=self.hdim+self.cdim, stride=self.feat_ratio, pretrained=pretrained, num_input_images=2)

    def upsample_depth(self, depth, mask, ratio=8, image_size=None):
        """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = depth.shape
        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(depth, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        # return up_flow.reshape(N, 1, ratio*H, ratio*W)

        up_flow = up_flow.reshape(N, 1, ratio*H, ratio*W)
        up_flow = F.interpolate(up_flow, size=image_size, mode='bilinear', align_corners=True)
        
        return up_flow
    
    def get_cost_each(self, pose, fmap, fmap_ref, depth, K, ref_K, scale_factor):
        """
            depth: (b, 1, h, w)
            fmap, fmap_ref: (b, c, h, w)
        """
        pose = Pose.from_vec(pose)

        device = depth.device
        cam = Camera(K=K.float()).scaled(scale_factor).to(device) # tcw = Identity
        ref_cam = Camera(K=ref_K.float(), Twc=pose).scaled(scale_factor).to(device)
        
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)

        fmap_warped = F.grid_sample(fmap_ref, ref_coords, mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
        cost = (fmap - fmap_warped)**2
        
        return cost
    
    def depth_cost_calc(self, inv_depth, fmap, fmaps_ref, pose_list, K, ref_K, scale_factor):
        cost_list = []
        for i, (pose, fmap_r) in enumerate(zip(pose_list, fmaps_ref)):
            cost = self.get_cost_each(pose, fmap, fmap_r, inv2depth(inv_depth), K, ref_K[i], scale_factor)
            cost_list.append(cost)  # (b, c, h, w)
        
        cost = torch.stack(cost_list, dim=1).mean(dim=1)
        return cost
    
    def forward(self, fmaps, target_image, ref_imgs, target_intrinsics, ref_intrinsics,
                min_depth=0.1, max_depth=100, scaled_shape=(378, 504)
                ):
        """ Estimate inv depth and poses """
        num_views = ref_imgs.shape[0]
        image_size = target_image.shape[-2:]
        scaled_shape[0] = scaled_shape[0].to(target_intrinsics.device)
        scaled_shape[1] = scaled_shape[1].to(target_intrinsics.device)

        if scaled_shape[0] > 0 and scaled_shape[1] > 0:
            target_image = F.interpolate(target_image, size=scaled_shape, mode='bilinear', align_corners=True)
            ref_imgs = F.interpolate(ref_imgs, size=scaled_shape, mode='bilinear', align_corners=True)
    
            target_intrinsics[..., 0] *= scaled_shape[0] / image_size[0]
            target_intrinsics[..., 1] *= scaled_shape[1] / image_size[1]
    
            ref_intrinsics[..., 0] *= scaled_shape[0] / image_size[0]
            ref_intrinsics[..., 1] *= scaled_shape[1] / image_size[1]

        if self.out_normalize:
            self.scale_inv_depth = partial(disp_to_depth, min_depth=min_depth, max_depth=max_depth)
        else:
            self.scale_inv_depth = lambda x: (x, None) # identity
        
        # run the feature network
        fmaps = self.fnet(torch.cat([target_image, ref_imgs], dim=0))
        # print(f'fmaps shape: {fmaps.shape}')

        fmaps = torch.split(fmaps, [target_image.shape[0]] * (1 + num_views), dim=0)
        # print(f'fmaps shape: {len(fmaps)}')
        fmap1, fmaps_ref = fmaps[0], fmaps[1:]
        # print(f'[DEBUG] fmap1 shape: {fmap1.shape}')
        
        # Initialize camera poses.
        pose_list_init = []
        for fmap_ref in fmaps_ref:
            pose_list_init.append(self.pose_head(torch.cat([fmap1, fmap_ref], dim=1)))
        
        # Initialize depths.
        inv_depth_init = self.depth_head(fmap1, act_fn=torch.sigmoid)
        # print(f'[DEBUG] inv_depth_init shape: {inv_depth_init.shape}')
        up_mask = self.upmask_net(fmap1)
        inv_depth_up_init = self.upsample_depth(inv_depth_init, up_mask, ratio=self.feat_ratio, image_size=image_size)

        inv_depth_predictions = [self.scale_inv_depth(inv_depth_up_init)[0]]
        pose_predictions = [[pose.clone() for pose in pose_list_init]]
        
        # run the context network for optimization
        if self.iters > 0:
            cnet_depth = self.cnet_depth(target_image)        
            hidden_d, inp_d = torch.split(cnet_depth, [self.hdim, self.cdim], dim=1)
            hidden_d = torch.tanh(hidden_d)
            inp_d = torch.relu(inp_d)
            
            img_pairs = []
            for i in range(num_views):
                ref_img = ref_imgs[i].unsqueeze(0)
                img_pairs.append(torch.cat([target_image, ref_img], dim=1))
            
            cnet_pose_list = self.cnet_pose(img_pairs)
            hidden_p_list, inp_p_list = [], []
            for cnet_pose in cnet_pose_list:
                hidden_p, inp_p = torch.split(cnet_pose, [self.hdim, self.cdim], dim=1)
                hidden_p_list.append(torch.tanh(hidden_p))
                inp_p_list.append(torch.relu(inp_p))
    
        pose_list = pose_list_init
        inv_depth = inv_depth_init
        inv_depth_up = None
        for itr in range(self.iters):
            inv_depth = inv_depth.detach()
            pose_list = [pose.detach() for pose in pose_list]

            # Calculate the feature cost map between each nearby view and the target view.
            pose_cost_func_list = []
            for i, fmap_ref in enumerate(fmaps_ref):
                pose_cost_func_list.append(partial(self.get_cost_each, fmap=fmap1, fmap_ref=fmap_ref,
                                                   depth=inv2depth(self.scale_inv_depth(inv_depth)[0]),
                                                   K=target_intrinsics, ref_K=ref_intrinsics[i], scale_factor=1.0/self.feat_ratio))

            depth_cost_func = partial(self.depth_cost_calc, fmap=fmap1, fmaps_ref=fmaps_ref,
                                      pose_list=pose_list, K=target_intrinsics,
                                      ref_K=ref_intrinsics, scale_factor=1.0/self.feat_ratio)

    
            #########  update depth ##########
            hidden_d, up_mask_seqs, inv_depth_seqs = self.update_block_depth(hidden_d, depth_cost_func,
                                                                             inv_depth, inp_d,
                                                                             seq_len=self.seq_len, 
                                                                             scale_func=self.scale_inv_depth)
            
            up_mask_seqs, inv_depth_seqs = [up_mask_seqs[-1]], [inv_depth_seqs[-1]]
            
            # upsample predictions
            for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
                inv_depth_up = self.upsample_depth(inv_depth_i, up_mask_i, ratio=self.feat_ratio, image_size=image_size)
                inv_depth_predictions.append(self.scale_inv_depth(inv_depth_up)[0])
            inv_depth = inv_depth_seqs[-1]
            
            #########  update pose ###########
            pose_list_seqs = [None] * len(pose_list)
            for i, (pose, hidden_p) in enumerate(zip(pose_list, hidden_p_list)):
                hidden_p, pose_seqs = self.update_block_pose(hidden_p, pose_cost_func_list[i],
                                                             pose, inp_p_list[i], seq_len=self.seq_len)
                hidden_p_list[i] = hidden_p

                pose_seqs = [pose_seqs[-1]]
                pose_list_seqs[i] = pose_seqs
                
            for pose_list_i in zip(*pose_list_seqs):
                pose_predictions.append([pose.clone() for pose in pose_list_i])

            pose_list = list(zip(*pose_list_seqs))[-1]
        
        if not self.training:
            return inv_depth_predictions[-1], \
                   torch.stack(pose_predictions[-1], dim=1).view(target_image.shape[0], len(ref_imgs), 6), fmap1 #(b, n, 6)
        
        return inv_depth_predictions, \
               torch.stack([torch.stack(poses_ref, dim=1) for poses_ref in pose_predictions], dim=2), fmap1 #(b, n, iters, 6)