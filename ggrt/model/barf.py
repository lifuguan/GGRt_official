import os

import numpy as np
import torch
import tqdm
from easydict import EasyDict as edict
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from perf.utils import utils, visualization
from perf.model.nerf import base
from perf.model.nerf import nerf
from perf.geometry import camera


class BARFTrainer(nerf.VanillaNeRFTrainer):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.trainer_name = 'BARFTrainer'
        self.pose_scheduler = None

    def build_networks(self, graph_model: base.BaseNeRFGraphModel, config):
        super().build_networks(graph_model, config)
        
        if config.camera.noise != None:
            # pre-generate synthetic pose perturbation
            se3_noise = torch.randn(
                len(self.train_data), 6, device=config.device) * config.camera.noise
            self.graph.pose_noise = camera.lie.se3_to_SE3(se3_noise)
        
        # TODO(chenyu): it's a little weird for this piece of code appears here,
        # may move it to BARF() later.
        self.graph.se3_refine = torch.nn.Embedding(
            len(self.train_data), 6).to(config.device)
        torch.nn.init.zeros_(self.graph.se3_refine.weight)

    def setup_optimizer(self, config):
        super().setup_optimizer(config)
        
        optimizer = getattr(torch.optim, config.optim.algo)
        self.pose_optimizer = optimizer([
            dict(params=self.graph.se3_refine.parameters(), lr=config.optim.lr_pose)
        ])
        
        # set up scheduler
        if config.optim.sched_pose:
            scheduler = getattr(torch.optim.lr_scheduler, config.optim.sched_pose.type)
            if config.optim.lr_pose_end:
                assert(config.optim.sched_pose.type == "ExponentialLR")
                config.optim.sched_pose.gamma = \
                    (config.optim.lr_pose_end / config.optim.lr_pose) ** (1. / config.max_iter)
            kwargs = {k:v for k, v in config.optim.sched_pose.items() if k != "type"}
            self.pose_scheduler = scheduler(self.pose_optimizer, **kwargs)

    def train_iteration(self, config, data, loader):
        self.pose_optimizer.zero_grad()
        
        if config.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.pose_optimizer.param_groups[0]["lr_orig"] = \
                self.pose_optimizer.param_groups[0]["lr"] # cache the original learning rate
            self.pose_optimizer.param_groups[0]["lr"] *= \
                min(1, self.iteration / config.optim.warmup_pose)
        
        loss = super().train_iteration(config, data, loader)
        self.pose_optimizer.step()
        
        if config.optim.warmup_pose:
            self.pose_optimizer.param_groups[0]["lr"] = \
                self.pose_optimizer.param_groups[0]["lr_orig"] # reset learning rate
        
        if config.optim.sched_pose:
            self.pose_scheduler.step()
        self.graph.nerf.progress.data.fill_(self.iteration / config.max_iter)
        
        if config.nerf.fine_sampling:
            self.graph.nerf_fine.progress.data.fill_(self.iteration / config.max_iter)
        return loss

    @torch.no_grad()
    def validate(self, config, ep=None):
        pose, pose_GT = self.get_all_training_poses(config)
        _, self.graph.sim3 = self.prealign_cameras(config, pose, pose_GT)
        
        super().validate(config, ep=ep)

    @torch.no_grad()
    def log_scalars(self, config, data, loss, writer, metric=None, step=0, split="train"):
        super().log_scalars(config, data, loss, writer, metric=metric, step=step, split=split)
        if split == "train":
            # log learning rate
            lr = self.pose_optimizer.param_groups[0]["lr"]
            writer.add_scalar(f"{split}/lr_pose", lr, step)
        
        # compute pose error
        if split == "train": # and config.data.dataset in ["blender", "llff"]:
            pose, pose_GT = self.get_all_training_poses(config)
            pose_aligned, _ = self.prealign_cameras(config, pose, pose_GT)
            error = self.evaluate_camera_alignment(config, pose_aligned, pose_GT)
            writer.add_scalar(f"{split}/error_R", error.R.mean(), step)
            writer.add_scalar(f"{split}/error_t", error.t.mean(), step)

    @torch.no_grad()
    def visualize(self, config, data, writer, step=0, split="train"):
        super().visualize(config, data, writer, step=step, split=split)

    def save_checkpoint(self, score: float = 0.0):
        self.ckpt_manager.save(
            model=self.graph,
            optimizers={'optim': self.optimizer, 'pose_optim': self.pose_optimizer},
            schedulers={'sched': self.scheduler, 'pose_sched': self.pose_scheduler},
            step=self.iteration,
            score=score
        )

    def load_checkpoint(self, config: DictConfig, mode='train'):
        self.iter_start = self.ckpt_manager.load(
            save_path=config.output_path,
            model=self.graph,
            optimizers={'optim': self.optimizer, 'pose_optim': self.pose_optimizer} if mode == 'train' else None,
            schedulers={'sched': self.scheduler, 'pose_sched': self.pose_scheduler} if self.scheduler is not None else None,
            device=config.device
        )

    @torch.no_grad()
    def get_all_training_poses(self, config):
        # Get ground-truth (canonical) camera poses.
        pose_GT = self.train_data.get_all_camera_poses(config).to(config.device)
        
        # Add synthetic pose perturbation to all training data.
        if config.data.dataset == "blender":
            pose = pose_GT
            if config.camera.noise:
                pose = camera.pose.compose([self.graph.pose_noise, pose])
        else:
            pose = self.graph.pose_eye
        # Add learned pose correction to all training data.
        pose_refine = camera.lie.se3_to_SE3(self.graph.se3_refine.weight)
        pose = camera.pose.compose([pose_refine, pose])
        return pose, pose_GT

    @torch.no_grad()
    def prealign_cameras(self, config, pose, pose_GT):
        # Compute 3D similarity transform via Procrustes analysis.
        center = torch.zeros(1, 1, 3, device=config.device)
        center_pred = camera.cam2world(center, pose)[:, 0] # [N,3]
        center_GT = camera.cam2world(center, pose_GT)[:, 0] # [N,3]
        try:
            sim3 = camera.procrustes_analysis(center_GT, center_pred)
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=config.device))
        
        # Align the camera poses.
        center_aligned = \
            (center_pred - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
        R_aligned = pose[..., :3] @ sim3.R.t()
        t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
        pose_aligned = camera.pose(R=R_aligned, t=t_aligned)
        return pose_aligned, sim3

    @torch.no_grad()
    def evaluate_camera_alignment(self, config, pose_aligned, pose_GT):
        # measure errors in rotation and translation
        R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
        R_GT,t_GT = pose_GT.split([3, 1], dim=-1)
        R_error = camera.rotation_distance(R_aligned, R_GT)
        t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
        error = edict(R=R_error, t=t_error)
        return error

    @torch.no_grad()
    def evaluate_full(self, config):
        self.graph.eval()
        # evaluate rotation/translation
        pose, pose_GT = self.get_all_training_poses(config)
        pose_aligned, self.graph.sim3 = self.prealign_cameras(config, pose, pose_GT)
        error = self.evaluate_camera_alignment(config,pose_aligned,pose_GT)
        print("--------------------------")
        print(f"rot:   {np.rad2deg(error.R.mean().cpu()):8.3f}")
        print(f"trans: {error.t.mean():10.5f}")
        print("--------------------------")
        
        # Dump numbers.
        quant_fname = f"{self.output_path}/quant_pose.txt"
        with open(quant_fname, "w") as file:
            for i, (err_R, err_t) in enumerate(zip(error.R, error.t)):
                file.write(f"{i} {err_R.item()} {err_t.item()}\n")
        
        # Evaluate novel view synthesis.
        super().evaluate_full(config)

    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self, config, var):
        # use another se3 Parameter to absorb the remaining pose errors
        var.se3_refine_test = torch.nn.Parameter(torch.zeros(1, 6, device=config.device))
        optimizer = getattr(torch.optim, config.optim.algo)
        optim_pose = optimizer([dict(params=[var.se3_refine_test], lr=config.optim.lr_pose)])
        iterator = tqdm.trange(
            config.optim.test_iter, desc="test-time optim.", leave=False, position=1)
        
        for it in iterator:
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.forward(config, var, mode="test-optim")
            loss = self.graph.compute_loss(config, var, mode="test-optim")
            loss = self.summarize_loss(config, var, loss)
            loss.all.backward()
            optim_pose.step()
            iterator.set_postfix(loss=f"{loss.all:.3f}")
        return var

    @torch.no_grad()
    def generate_videos_pose(self, config):
        self.graph.eval()
        fig = plt.figure(
            figsize=(10, 10) if config.data.dataset == "blender" else (16, 8))
        cam_path = f"{self.output_path}/poses"
        os.makedirs(cam_path, exist_ok=True)
        ep_list = []
        
        for ep in range(0, config.max_iter + 1, config.freq.ckpt):
            # load checkpoint (0 is random init)
            if ep != 0:
                try:
                    utils.restore_checkpoint(config, self, resume=ep)
                except:
                    continue
            
            # get the camera poses
            pose, pose_ref = self.get_all_training_poses(config)
            if config.data.dataset in ["blender", "llff"]:
                pose_aligned, _ = self.prealign_cameras(config, pose, pose_ref)
                pose_aligned, pose_ref = pose_aligned.detach().cpu(), pose_ref.detach().cpu()
                dict(
                    blender=visualization.plot_save_poses_blender,
                    llff=visualization.plot_save_poses,
                )[config.data.dataset](
                    config, fig, pose_aligned, pose_ref=pose_ref, path=cam_path, ep=ep
                )
            else:
                pose = pose.detach().cpu()
                visualization.plot_save_poses(
                    config, fig, pose, pose_ref=None, path=cam_path, ep=ep
                )
            ep_list.append(ep)
        plt.close()
        
        # Write videos.
        print("writing videos...")
        list_fname = f"{cam_path}/temp.list"
        with open(list_fname, "w") as file:
            for ep in ep_list:
                file.write(f"file {ep}.png\n")
        cam_vid_fname = f"{self.output_path}/poses.mp4"
        os.system(f"ffmpeg -y -r 30 -f concat -i {list_fname} " + \
                f"-pix_fmt yuv420p {cam_vid_fname} >/dev/null 2>&1")
        os.remove(list_fname)


class BARFGraphModel(nerf.VanillaNeRFGraphModel):
    def __init__(self, config):
        super().__init__(config)

        self.nerf = BARF(config)
        self.graph_model_name = 'BARFGraphModel'

        if config.nerf.fine_sampling:
            self.nerf_fine = BARF(config)
        self.pose_eye = torch.eye(3, 4).to(config.device)

    def get_pose(self, config, data, mode=None):
        if mode == "train":
            # add the pre-generated pose perturbations
            if config.data.dataset == "blender":
                if config.camera.noise:
                    data.pose_noise = self.pose_noise[data.idx]
                    pose = camera.pose.compose([data.pose_noise, data.pose])
                else:
                    pose = data.pose
            elif config.data.dataset == 'llff':
                pose = self.pose_eye
            else:
                data.pose_noise = self.pose_noise[data.idx]
                pose = camera.pose.compose([data.pose_noise, data.pose])
            
            # Add learnable pose correction.
            data.se3_refine = self.se3_refine.weight[data.idx]
            pose_refine = camera.lie.se3_to_SE3(data.se3_refine)
            pose = camera.pose.compose([pose_refine, pose])
        elif mode in ["val", "eval", "test-optim"]:
            # Align test pose to refined coordinate system (up to sim3)
            sim3 = self.sim3
            center = torch.zeros(1, 1, 3, device=config.device)
            center = camera.cam2world(center, data.pose)[:, 0] # [N,3]
            center_aligned = (center - sim3.t0) / sim3.s0 @ sim3.R * sim3.s1 + sim3.t1
            R_aligned = data.pose[..., :3] @ self.sim3.R
            t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
            pose = camera.pose(R=R_aligned, t=t_aligned)
            
            # Additionally factorize the remaining pose imperfection.
            if config.optim.test_photo and mode != "val":
                pose = camera.pose.compose([data.pose_refine_test, pose])
        else:
            pose = data.pose
        return pose


class BARF(nerf.VanillaNeRF):
    def __init__(self, config):
        super().__init__(config)

        self.nerf_name = 'BARF'
        
        # use Parameter so it could be checkpointed
        self.progress = torch.nn.Parameter(torch.tensor(0.))

        # If enable coarse to fine positional encoding.
        self.enable_coarse_to_fine = True

    def positional_encoding(self, config, input, L): # [B,...,N]
        input_enc = super().positional_encoding(config, input, L=L) # [B,...,2NL]
        
        if not self.enable_coarse_to_fine:
            return input_enc
        
        # coarse-to-fine: smoothly mask positional encoding for BARF.
        if config.barf_c2f is not None:
            # Set weights for different frequency bands.
            start, end = config.barf_c2f
            alpha = (self.progress.data - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=config.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            
            # Apply weights.
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L) * weight).view(*shape)
        
        return input_enc

    def activate_coarse_to_fine(self):
        self.enable_coarse_to_fine = True

    def deactivate_coarse_to_fine(self):
        self.enable_coarse_to_fine = False
