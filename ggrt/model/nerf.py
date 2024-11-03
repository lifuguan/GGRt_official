import os
import time

import numpy as np
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as torchvision_F
import tqdm

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from easydict import EasyDict as edict
from omegaconf import DictConfig

import lpips
import tinycudann as tcnn

from perf.model.base_trainer import BaseTrainer
from perf.model import ssim_torch
from perf.model.nerf import base
from perf.geometry import camera
from perf.utils import utils, visualization


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply


class VanillaNeRFGraphModel(base.BaseNeRFGraphModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        self.nerf = VanillaNeRF(config)
        # self.nerf = TCNNVanillaNeRF(config)
        self.graph_model_name = "VanillaNeRFGraphModel"
        if config.nerf.fine_sampling:
            self.nerf_fine = VanillaNeRF(config)
            # self.nerf_fine = TCNNVanillaNeRF(config)

    def forward(self, config: DictConfig, var, mode=None):
        batch_size = len(var.idx)
        
        # NOTE: for datasets without known camera poses, the validation won't
        # be able to get the desired results.
        pose = self.get_pose(config, var, mode=mode)
        
        # render images
        if config.nerf.rand_rays and mode in ["train", "test-optim"]:
            # sample random rays for optimization
            var.ray_idx = torch.randperm(
                config.H * config.W, device=config.device
            )[:config.nerf.rand_rays // batch_size]
            ret = self.render(
                config, pose, intr=var.intr, ray_idx=var.ray_idx, mode=mode
            ) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(
                config, pose, intr=var.intr, mode=mode
            ) if config.nerf.rand_rays else self.render(
                config, pose, intr=var.intr, mode=mode
            ) # [B,HW,3],[B,HW,1]
        
        var.update(ret)
        
        return var

    def compute_loss(self, config: DictConfig, var, mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size, 3, config.H * config.W).permute(0, 2, 1)
        
        if config.nerf.rand_rays and mode in ["train", "test-optim"]:
            image = image[:, var.ray_idx]
        
        # compute image losses
        if config.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb, image)
        
        if config.loss_weight.render_fine is not None:
            assert(config.nerf.fine_sampling)
            loss.render_fine = self.MSE_loss(var.rgb_fine, image)
        
        return loss

    def get_pose(self, config: DictConfig, data, mode=None):
        return data.pose

    def render(self, config: DictConfig, pose, intr=None, ray_idx=None, mode=None, **kwargs):
        batch_size = len(pose)
        center, ray = camera.get_center_and_ray(config, pose, intr=intr) # [B,HW,3]
        while ray.isnan().any():
            # TODO: weird bug, ray becomes NaN arbitrarily if batch_size > 1,
            # not deterministic reproducible
            center, ray = camera.get_center_and_ray(config, pose, intr=intr) # [B,HW,3]
        
        if ray_idx is not None:
            # consider only subset of rays
            center, ray = center[:, ray_idx], ray[:, ray_idx]
        
        if config.camera.ndc:
            # convert center/ray representations to NDC
            center, ray = camera.convert_NDC(config, center, ray, intr=intr)

        # render with main MLP
        depth_samples = self.sample_depth(
            config, batch_size, num_rays=ray.shape[1]) # [B,HW,N,1]
        
        rgb_samples, density_samples = self.nerf.forward_samples(
            config, center, ray, depth_samples, mode=mode)
        
        rgb, depth, opacity, prob = self.nerf.composite(
            config, ray, rgb_samples, density_samples, depth_samples)
        
        ret = edict(rgb=rgb, depth=depth, opacity=opacity) # [B,HW,K]
        
        # render with fine MLP from coarse MLP
        if config.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth according to coarse empirical distribution.
                depth_samples_fine = self.sample_depth_from_pdf(
                    config, pdf=prob[..., 0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([
                    depth_samples,depth_samples_fine
                ], dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            
            rgb_samples, density_samples = self.nerf_fine.forward_samples(
                config, center, ray, depth_samples, mode=mode)
            rgb_fine, depth_fine, opacity_fine,_ = self.nerf_fine.composite(
                config, ray, rgb_samples, density_samples, depth_samples)
            ret.update(
                rgb_fine=rgb_fine, depth_fine=depth_fine, opacity_fine=opacity_fine
            ) # [B,HW,K]
        
        return ret

    def render_by_slices(self, config: DictConfig, pose, intr=None, mode=None):
        ret_all = edict(rgb=[], depth=[], opacity=[])
        if config.nerf.fine_sampling:
            ret_all.update(rgb_fine=[], depth_fine=[], opacity_fine=[])
        
        # render the image by slices for memory considerations
        for c in range(0, config.H * config.W, config.nerf.rand_rays):
            ray_idx = torch.arange(
                c, min(c + config.nerf.rand_rays, config.H * config.W
            ), device=config.device)
            
            ret = self.render(
                config, pose, intr=intr, ray_idx=ray_idx, mode=mode) # [B,R,3],[B,R,1]
            
            for k in ret:
                ret_all[k].append(ret[k])
        
        # group all slices of images
        for k in ret_all:
            ret_all[k] = torch.cat(ret_all[k], dim=1)
        return ret_all

    def sample_depth(self, config: DictConfig, batch_size, num_rays=None):
        depth_min, depth_max = config.nerf.depth.range
        num_rays = num_rays or config.H * config.W
        
        rand_samples = torch.rand(
            batch_size, num_rays, config.nerf.sample_intvs, 1, device=config.device
        ) if config.nerf.sample_stratified else 0.5
        rand_samples += torch.arange(
            config.nerf.sample_intvs, device=config.device
        )[None, None, :, None].float() # [B,HW,N,1]
        
        depth_samples = rand_samples / config.nerf.sample_intvs * (
            depth_max - depth_min) + depth_min # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[config.nerf.depth.param]
        
        return depth_samples

    def sample_depth_from_pdf(self, config: DictConfig, pdf):
        depth_min, depth_max = config.nerf.depth.range
        
        # Get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [B,HW,N+1]
        
        # Take uniform samples
        grid = torch.linspace(
            0, 1, config.nerf.sample_intvs_fine + 1, device=config.device
        ) # [Nf+1]
        unif = 0.5 * (grid[:-1] + grid[1:]).repeat(*cdf.shape[:-1], 1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf, unif, right=True) # [B,HW,Nf] \in {1...N}
        
        # Inverse transform sampling from CDF.
        depth_bin = torch.linspace(
            depth_min, depth_max, config.nerf.sample_intvs + 1, device=config.device
        ) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1], 1) # [B, HW, N+1]
        depth_low = depth_bin.gather(dim=2, index=(idx-1).clamp(min=0)) # [B, HW, Nf]
        depth_high = depth_bin.gather(
            dim=2, index=idx.clamp(max=config.nerf.sample_intvs
        )) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2, index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2, index=idx.clamp(max=config.nerf.sample_intvs)) # [B,HW,Nf]
        
        # Linear interpolation.
        t = (unif - cdf_low) / (cdf_high - cdf_low + 1e-8) # [B,HW,Nf]
        depth_samples = depth_low + t * (depth_high - depth_low) # [B,HW,Nf]
        
        return depth_samples[...,None] # [B,HW,Nf,1]


class VanillaNeRFTrainer(BaseTrainer):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.trainer_name = 'VanillaNeRFTrainer'
        self.lpips_loss = lpips.LPIPS(net="alex").to(config.device)

    def load_dataset(self, config: DictConfig, eval_split="val"):
        super().load_dataset(config, eval_split=eval_split)
        
        # prefetch all training data
        self.train_data.prefetch_all_data(config)
        self.train_data.all = edict(
            utils.move_to_device(self.train_data.all, config.device)
        )

    def setup_optimizer(self, config: DictConfig):
        self.logger.info("setting up optimizers...")
        optimizer = getattr(torch.optim, config.optim.algo)
        self.optimizer = optimizer([
            dict(params=self.graph.nerf.parameters(), lr=config.optim.lr)
        ])
        # self.optimizer = optimizer(
        #     filter(lambda p: p.requires_grad, self.graph.nerf.parameters()),
        #     lr=config.optim.lr
        # )

        if config.nerf.fine_sampling:
            self.optimizer.add_param_group(
                dict(params=self.graph.nerf_fine.parameters(), lr=config.optim.lr)
            )
            # self.optimizer.add_param_group(
            #     filter(lambda p: p.requires_grad, self.graph.nerf.parameters()),
            #     lr=config.optim.lr
            # )
        
        # set up scheduler
        if config.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, config.optim.sched.type)
            if config.optim.lr_end:
                assert(config.optim.sched.type == "ExponentialLR")
                config.optim.sched.gamma = \
                    (config.optim.lr_end / config.optim.lr) ** (1. / config.max_iter)
            
            kwargs = {k:v for k,v in config.optim.sched.items() if k != "type"}
            self.scheduler = scheduler(self.optimizer, **kwargs)

    def train(self, config: DictConfig):
        # before training
        self.logger.title("Training Start...")
        self.timer = edict(start=time.time(), it_mean=None)
        self.graph.train()
        self.epoch = 0 # dummy for timer
        
        if config.resume is not None:
            self.logger.info('Resuming from latest checkpoint...')
            self.load_checkpoint(config)
        # self.iteration = self.iter_start
        
        loader = tqdm.trange(config.max_iter, desc="training", leave=False)
        for self.iteration in loader:
            if self.iteration < self.iter_start:
                continue
            
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(config, var, loader)
            if config.optim.sched:
                self.scheduler.step()
            
            if (self.iteration + 1) % config.freq.val == 0:
                self.validate(config, self.iteration)
            
            if (self.iteration + 1) % config.freq.ckpt == 0:
                # score = self.validate(config, self.iteration)
                self.save_checkpoint(score=0.0)
        
        # after training
        self.train_writer.flush()
        self.train_writer.close()
        self.val_writer.flush()
        self.val_writer.close()

        self.logger.title("Training Done!")

    @torch.no_grad()
    def log_scalars(self, config, var, loss, writer, metric=None, step=0, split="train"):
        super().log_scalars(config, var, loss, writer, metric=metric, step=step, split=split)
        # log learning rate
        if split == "train":
            lr = self.optimizer.param_groups[0]["lr"]
            writer.add_scalar(f"{split}/lr", lr, step)
            if config.nerf.fine_sampling:
                lr = self.optimizer.param_groups[1]["lr"]
                writer.add_scalar(f"{split}/lr_fine", lr, step)
        
        # compute PSNR
        psnr = -10 * loss.render.log10()
        writer.add_scalar(f"{split}/PSNR", psnr, step)
        if config.nerf.fine_sampling:
            psnr = -10 * loss.render_fine.log10()
            writer.add_scalar(f"{split}/PSNR_fine", psnr, step)

    @torch.no_grad()
    def visualize(self, config: DictConfig, data, writer, step=0, split="train", eps=1e-10):
        if config.tb:
            visualization.tb_image(config, writer, step, split, "image", data.image)
            if not config.nerf.rand_rays or split != "train":
                invdepth = (1 - data.depth) / data.opacity \
                    if config.camera.ndc else 1 / (data.depth / data.opacity + eps)
                rgb_map = data.rgb.view(-1, config.H, config.W, 3).permute(0, 3, 1, 2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1, config.H, config.W, 1).permute(0, 3, 1, 2) # [B,1,H,W]
                
                visualization.tb_image(config, writer, step, split, "rgb", rgb_map)
                visualization.tb_image(config, writer, step, split, "invdepth", invdepth_map)
                
                if config.nerf.fine_sampling:
                    invdepth = (1 - data.depth_fine) / data.opacity_fine \
                        if config.camera.ndc else 1 / (data.depth_fine / data.opacity_fine + eps)
                    rgb_map = data.rgb_fine.view(
                        -1, config.H, config.W, 3).permute(0, 3, 1, 2) # [B,3,H,W]
                    invdepth_map = invdepth.view(
                        -1, config.H, config.W, 1).permute(0, 3, 1, 2) # [B,1,H,W]
                    visualization.tb_image(config, writer, step, split, "rgb_fine", rgb_map)
                    visualization.tb_image(config, writer, step, split, "invdepth_fine", invdepth_map)

    @torch.no_grad()
    def get_all_training_poses(self, config: DictConfig):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(config).to(config.device)
        return None, pose_GT

    @torch.no_grad()
    def evaluate_full(self, config: DictConfig, eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)
        res = []
        test_path = f"{self.output_path}/test_view"
        os.makedirs(test_path, exist_ok=True)
        
        for i, batch in enumerate(loader):
            var = edict(batch)
            var = utils.move_to_device(var, config.device)
            if config.model == "barf" and config.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized
                # poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(config, var)
            var = self.graph.forward(config, var, mode="eval")
            
            # evaluate view synthesis
            invdepth = (1 - var.depth) / var.opacity \
                if config.camera.ndc else 1 / (var.depth / var.opacity + eps)
            rgb_map = var.rgb.view(-1, config.H, config.W, 3).permute(0, 3, 1, 2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1, config.H, config.W, 1).permute(0, 3, 1, 2) # [B,1,H,W]
            psnr = -10 * self.graph.MSE_loss(rgb_map, var.image).log10().item()
            ssim = ssim_torch.ssim(rgb_map, var.image).item()
            lpips = self.lpips_loss(rgb_map * 2 - 1, var.image * 2 - 1).item()
            res.append(edict(psnr=psnr, ssim=ssim, lpips=lpips))
            
            # Dump novel views.
            torchvision_F.to_pil_image(
                rgb_map.cpu()[0]).save(f"{test_path}/rgb_{i}.png")
            torchvision_F.to_pil_image(
                var.image.cpu()[0]).save(f"{test_path}/rgb_GT_{i}.png")
            torchvision_F.to_pil_image(
                invdepth_map.cpu()[0]).save(f"{test_path}/depth_{i}.png")
        
        # Show results in terminal.
        print("--------------------------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")
        
        # Dump numbers to file.
        quant_fname = f"{self.output_path}/quant.txt"
        with open(quant_fname, "w") as file:
            for i, r in enumerate(res):
                file.write(f"{i} {r.psnr} {r.ssim} {r.lpips}\n")

    @torch.no_grad()
    def generate_videos_synthesis(self, config, eps=1e-10):
        self.graph.eval()
        if config.data.dataset == "blender":
            test_path = f"{self.output_path}/test_view"
            # Assume the test view synthesis are already generated.
            print("writing videos...")
            rgb_vid_fname = f"{self.output_path}/test_view_rgb.mp4"
            depth_vid_fname = f"{self.output_path}/test_view_depth.mp4"
            os.system(f"ffmpeg -y -framerate 30 -i {test_path}/rgb_%d.png " + \
                      f"-pix_fmt yuv420p {rgb_vid_fname} >/dev/null 2>&1")
            os.system(f"ffmpeg -y -framerate 30 -i {test_path}/depth_%d.png " + \
                      f"-pix_fmt yuv420p {depth_vid_fname} >/dev/null 2>&1")
        else:
            pose_pred, pose_GT = self.get_all_training_poses(config)
            poses = pose_pred if config.model == "barf" else pose_GT
            if config.model == "barf" and config.data.dataset == "llff":
                _, sim3 = self.prealign_cameras(config, pose_pred, pose_GT)
                scale = sim3.s1 / sim3.s0
            else:
                scale = 1
            
            # Rotate novel views around the "center" camera of all poses.
            idx_center = (
                poses - poses.mean(dim=0, keepdim=True)
            )[..., 3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(
                config, poses[idx_center], N=60, scale=scale).to(config.device)
            
            # Render the novel views.
            novel_path = f"{self.output_path}/novel_view"
            os.makedirs(novel_path, exist_ok=True)
            pose_novel_tqdm = tqdm.tqdm(
                pose_novel, desc="rendering novel views", leave=False)
            intr = edict(
                next(iter(self.test_loader))).intr[:1].to(config.device) # grab intrinsics
            for i,pose in enumerate(pose_novel_tqdm):
                ret = self.graph.render_by_slices(
                    config,pose[None], intr=intr) if config.nerf.rand_rays else \
                      self.graph.render(config, pose[None], intr=intr)
                invdepth = (1 - ret.depth) / ret.opacity \
                    if config.camera.ndc else 1/(ret.depth/ret.opacity+eps)
                rgb_map = ret.rgb.view(-1, config.H, config.W, 3).permute(0, 3, 1, 2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1, config.H, config.W, 1).permute(0, 3, 1, 2) # [B,1,H,W]
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save(f"{novel_path}/rgb_{i}.png")
                torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save(f"{novel_path}/depth_{i}.png")
            
            # Write videos.
            print("writing videos...")
            rgb_vid_fname = f"{self.output_path}/novel_view_rgb.mp4"
            depth_vid_fname = f"{self.output_path}/novel_view_depth.mp4"
            os.system(f"ffmpeg -y -framerate 30 -i {novel_path}/rgb_%d.png " + \
                      f"-pix_fmt yuv420p {rgb_vid_fname} >/dev/null 2>&1")
            os.system(f"ffmpeg -y -framerate 30 -i {novel_path}/depth_%d.png " + \
                      f"-pix_fmt yuv420p {depth_vid_fname} >/dev/null 2>&1")


class VanillaNeRF(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.nerf_name = 'VanillaNeRF'
        self.define_network(config)

    def define_network(self, config):
        input_3D_dim = 3 + 6 * config.arch.posenc.L_3D if config.arch.posenc else 3
        if config.nerf.view_dep:
            input_view_dim = \
                3 + 6 * config.arch.posenc.L_view if config.arch.posenc else 3
        
        # point-wise feature
        self.mlp_feat = torch.nn.ModuleList()
        L = utils.get_layer_dims(config.arch.layers_feat)
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = input_3D_dim
            if li in config.arch.skip:
                k_in += input_3D_dim
            if li == len(L) - 1:
                k_out += 1
            
            linear = torch.nn.Linear(k_in, k_out)
            if config.arch.tf_init:
                self.tensorflow_init_weights(
                    config, linear, out="first" if li == len(L) - 1 else None)
            self.mlp_feat.append(linear)
        
        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = utils.get_layer_dims(config.arch.layers_rgb)
        feat_dim = config.arch.layers_feat[-1]
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = feat_dim + (input_view_dim if config.nerf.view_dep else 0)
            linear = torch.nn.Linear(k_in, k_out)
            if config.arch.tf_init:
                self.tensorflow_init_weights(
                    config, linear, out="all" if li == len(L) - 1 else None)
            self.mlp_rgb.append(linear)

    def tensorflow_init_weights(self, config, linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out == "all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out == "first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:], gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight, gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, config, points_3D, ray_unit=None, mode=None): # [B,...,3]
        if config.arch.posenc:
            points_enc = self.positional_encoding(
                config, points_3D, L=config.arch.posenc.L_3D)
            points_enc = torch.cat([points_3D, points_enc], dim=-1) # [B,...,6L+3]
        else:
            points_enc = points_3D
        feat = points_enc
        
        # extract coordinate-based features
        for li, layer in enumerate(self.mlp_feat):
            if li in config.arch.skip:
                feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
            if li == len(self.mlp_feat) - 1:
                density = feat[..., 0]
                if config.nerf.density_noise_reg and mode == "train":
                    density += torch.randn_like(density) * config.nerf.density_noise_reg
                
                # relu_,abs_,sigmoid_,exp_....
                density_activ = getattr(torch_F, config.arch.density_activ)
                density = density_activ(density)
                feat = feat[..., 1:]
            feat = torch_F.relu(feat)
        
        # Predict RGB values.
        if config.nerf.view_dep:
            assert(ray_unit is not None)
            if config.arch.posenc:
                ray_enc = self.positional_encoding(
                    config, ray_unit, L=config.arch.posenc.L_view)
                ray_enc = torch.cat([ray_unit, ray_enc], dim=-1) # [B,...,6L+3]
            else:
                ray_enc = ray_unit
            feat = torch.cat([feat,ray_enc], dim=-1)
        
        for li, layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            if li != len(self.mlp_rgb) - 1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,...,3]

        return rgb, density

    def forward_samples(self, config, center, ray, depth_samples, mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(
            config, center, ray, depth_samples, multi_samples=True) # [B,HW,N,3]
        if config.nerf.view_dep:
            ray_unit = torch_F.normalize(ray, dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[..., None,:].expand_as(
                points_3D_samples) # [B,HW,N,3]
        else:
            ray_unit_samples = None
        
        rgb_samples,density_samples = self.forward(
            config, points_3D_samples, ray_unit=ray_unit_samples, mode=mode
        ) # [B,HW,N],[B,HW,N,3]
        return rgb_samples,density_samples

    def composite(self, config, ray, rgb_samples, density_samples, depth_samples):
        ray_length = ray.norm(dim=-1, keepdim=True) # [B,HW,1]
        
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([
            depth_intv_samples, torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)
        ], dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples * ray_length # [B,HW,N]
        sigma_delta = density_samples * dist_samples # [B,HW,N]
        alpha = 1 - (-sigma_delta).exp_() # [B,HW,N]
        T = (-torch.cat([
            torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]
        ], dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
        prob = (T * alpha)[...,None] # [B,HW,N,1]
        
        # integrate RGB and depth weighted by probability
        depth = (depth_samples * prob).sum(dim=2) # [B,HW,1]
        rgb = (rgb_samples * prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        if config.nerf.setbg_opaque:
            rgb = rgb + config.data.bgcolor * (1 - opacity)
        return rgb, depth, opacity, prob # [B,HW,K]

    def positional_encoding(self, config, input, L): # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(
            L, dtype=torch.float32, device=config.device
        ) * np.pi # [L]
        spectrum = input[..., None] * freq # [B,...,N,L]
        sin,cos = spectrum.sin(), spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1) # [B,...,2NL]
        return input_enc


class TCNNVanillaNeRF(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.nerf_name = 'TCNNVanillaNeRF'

        self.geo_feat_dim = 15
        self.num_layers = 8 #2
        self.hidden_dim = 64
        self.num_layers_color = 3
        self.hidden_dim_color = 64

        self.define_network(config)

    def define_network(self, config):
        # sigma network.
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": config.arch.posenc.L_3D,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims, #32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": config.arch.layers_feat[1], # self.hidden_dim,
                "h_hidden_layers": self.num_layers - 1,
            },
        )

        # color network.
        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": config.arch.posenc.L_view,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": config.arch.layers_rgb[1], # self.hidden_dim_color,
                "n_hidden_layers": self.num_layers_color - 1
            }
        )

    def forward(self, config, points_3D, ray_unit=None, mode=None): # [B,...,3]
        num_batches, num_rays, num_samples = ray_unit.shape[0], ray_unit.shape[1], ray_unit.shape[2]
        points_3D = points_3D.reshape(num_batches * num_rays * num_samples, 3)
        ray_unit = ray_unit.reshape(num_batches * num_rays * num_samples, 3)

        # sigma
        x = self.encoder(points_3D)
        h = self.sigma_net(x)

        density = trunc_exp(h[..., 0]).reshape(num_batches, num_rays, num_samples)
        geo_feat = h[..., 1:]

        # color
        d = (ray_unit + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        rgb = torch.sigmoid(h).reshape(num_batches, num_rays, num_samples, 3)

        return rgb, density

    def forward_samples(self, config, center, ray, depth_samples, mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(
            config, center, ray, depth_samples, multi_samples=True) # [B,HW,N,3]
        
        if config.nerf.view_dep:
            ray_unit = torch_F.normalize(ray, dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[..., None,:].expand_as(
                points_3D_samples) # [B,HW,N,3]
        else:
            ray_unit_samples = None
        
        rgb_samples, density_samples = self.forward(
            config, points_3D_samples, ray_unit=ray_unit_samples, mode=mode
        ) # [B,HW,N],[B,HW,N,3]
        
        return rgb_samples, density_samples

    def composite(self, config, ray, rgb_samples, density_samples, depth_samples):
        ray_length = ray.norm(dim=-1, keepdim=True) # [B,HW,1]
        
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([
            depth_intv_samples, torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)
        ], dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples * ray_length # [B,HW,N]
        sigma_delta = density_samples * dist_samples # [B,HW,N]
        alpha = 1 - (-sigma_delta).exp_() # [B,HW,N]
        T = (-torch.cat([
            torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]
        ], dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
        prob = (T * alpha)[...,None] # [B,HW,N,1]
        
        # integrate RGB and depth weighted by probability
        depth = (depth_samples * prob).sum(dim=2) # [B,HW,1]
        rgb = (rgb_samples * prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        if config.nerf.setbg_opaque:
            rgb = rgb + config.data.bgcolor * (1 - opacity)
        return rgb, depth, opacity, prob # [B,HW,K]

    def positional_encoding(self, config, input, L): # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(
            L, dtype=torch.float32, device=config.device
        ) * np.pi # [L]
        spectrum = input[..., None] * freq # [B,...,N,L]
        sin,cos = spectrum.sin(), spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1) # [B,...,2NL]
        return input_enc
