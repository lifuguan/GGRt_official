import os
import random
import time
import tqdm
import socket
import visdom

import torch
import torch.distributed as dist
import numpy as np

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ggrt.base.checkpoint_manager import CheckPointManager
from ggrt.data_loaders import dataset_dict
from ggrt.data_loaders.create_training_dataset import create_training_dataset
from utils_loc import cycle


def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname,port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    
    return is_open


def seed_worker(worker_id):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseTrainer(object):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.trainer_name = 'BaseTrainer'
        self.config = config
        self.device = f"cuda:{config.local_rank}"

        self.output_path = os.path.join(config.rootdir, 'out', config.expname)
        if self.config.local_rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
            print(f'[INFO] Outputs will be saved to {self.output_path}')
        
        self.log_file = open(os.path.join(self.output_path, 'log.txt'), 'w')
        self.scheduler = None
        self.model = None
        self.scalars_to_log = dict()
        self.ckpt_manager = CheckPointManager(
            save_path=self.output_path,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5
        )
        
        self.train_done = False
        self._setup_visualizer()
        self._load_dataset()

        # Functions need to be overwritten.
        self.build_networks()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.compose_state_dicts()

        if self.config.local_rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
            print(f'[INFO] Outputs will be saved to {self.output_path}')
        
        # Check the validity of the model trainer.
        self._check()

    def __del__(self):
        if not self.train_done:
            score = self.validate()
            self.save_checkpoint(score=score)
        
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        
        self.log_file.close()

    def _check(self):
        assert self.train_dataset is not None
        assert self.train_loader is not None
        assert self.val_dataset is not None
        assert self.val_loader is not None
        assert self.model is not None
        assert os.path.exists(self.output_path) is True

        if self.config.distributed:
            assert self.train_sampler is not None

        if self.config.enable_tensorboard and self.config.local_rank == 0:
            assert self.writer is not None
        
        if self.config.enable_visdom and self.config.local_rank == 0:
            assert self.visdom is not None

    def _load_dataset(self):
        print(f'[INFO] Loading dataset...')
        # Create training dataset.
        self.train_dataset, self.train_sampler = create_training_dataset(self.config)
        # Currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1,
                                                   worker_init_fn=seed_worker,
                                                   num_workers=self.config.workers,
                                                   pin_memory=True,
                                                   sampler=self.train_sampler,
                                                #    shuffle = False)
                                                   shuffle=True if self.train_sampler is None else False)

        # Create validation dataset.
        self.val_dataset = dataset_dict[self.config.eval_dataset](self.config, 'validation',
                                                      scenes=self.config.eval_scenes)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)
        self.val_loader_iterator = iter(cycle(self.val_loader))

    def _setup_visualizer(self):
        print('[INFO] Setting up visualizers...', file=self.log_file)
        self.writer = None
        self.visdom = None

        # Setup tensorboard.
        if self.config.enable_tensorboard and self.config.local_rank == 0:
            log_dir = os.path.join(self.config.rootdir, 'logs', self.config.expname)
            self.writer = SummaryWriter(log_dir)
            print(f'[INFO] Saving tensorboard files to {log_dir}.')
        
        # Setup visdom.
        if self.config.enable_visdom and self.config.local_rank == 0:
            is_open = check_socket_open(self.config.visdom_server, self.config.visdom_port) # check if visdom server is runninng
            retry = None
            while not is_open:
                retry = input(f"visdom port ({self.config.visdom_port}) not open, retry? (y/n) ")
                if retry not in ["y", "n"]:
                    continue
                if retry == "y":
                    is_open = check_socket_open(self.config.visdom_server, self.config.visdom_port)
                else:
                    break

            self.visdom = visdom.Visdom(server=self.config.visdom_server, port=self.config.visdom_port, env='dbarf')
            print(f'[INFO] Visualizing camera poses at {self.config.visdom_server}:{self.config.visdom_port}')

    def build_networks(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def setup_optimizer(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def setup_loss_functions(self):
        """
            Implement this function.
        """
        raise NotImplementedError
    

    def demo(self):
        assert self.train_loader is not None
        assert self.val_loader is not None
        self.iteration =0
        iter_start = self.load_checkpoint(load_optimizer=not self.config.no_load_opt,
                                          load_scheduler=not self.config.no_load_scheduler)
        print(f'train set len {len(self.train_loader)}')
        lens=len(self.train_loader)
        for batch in self.train_loader:
            # Main training logic.
 
            self.demorender(data_batch=batch)
            
            
            self.iteration += 1

        self.train_done = True

    def train(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        pbar = tqdm.trange(self.config.n_iters, desc=f"Training {self.config.expname}", leave=False)

        iter_start = self.load_checkpoint(load_optimizer=not self.config.no_load_opt,
                                          load_scheduler=not self.config.no_load_scheduler)
        # if iter_start == 0:
        #     iter_start = int(self.config.ckpt_path[-10:-4])
        
        if self.config.distributed:
            # NOTE: Distributed mode can only be activated after loading models.
            self.model.to_distributed()
        
        self.epoch  = 0
        self.iteration = 0

        while self.iteration < iter_start:
            pbar.update(1)
            self.iteration += 1
            
        print(f'train set len {len(self.train_loader)}')
        while self.iteration < self.config.n_iters + 1:
            for self.train_data in self.train_loader:
                if self.config.distributed:
                    self.train_sampler.set_epoch(self.epoch)
                
                # Main training logic.
                self.train_iteration(data_batch=self.train_data)

                if self.config.local_rank == 0:
                    # Main validation logic.
                    # if self.iteration % self.config.n_validation == 0:
                    #     score = self.validate()
                    
                    # log to tensorboard.
                    if self.iteration % self.config.n_tensorboard == 0:
                        self.log_info()

                    # save checkpoint.
                    if self.iteration % self.config.n_checkpoint == 0:
                        score = self.validate()
                        self.save_checkpoint(score=score)
                
                pbar.update(1)
                
                self.iteration += 1
                if self.iteration > self.config.n_iters + 1:
                    break
            self.epoch += 1
        
        self.train_done = True

    def train_iteration(self, batch) -> None:
        raise NotImplementedError

    def demorender(self, data_batch) :
        raise NotImplementedError

    def demorender(self, data_batch) :
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> float:
        score = 0.
        """
            self.model.switch_to_eval()
            ... (implement validation logic here)
            self.model.switch_to_train()
        """
        
        return score

    def compose_state_dicts(self) -> None:
        """
            Implement this function and follow the format below:
            self.state_dicts = {'models': None, 'optimizers': None, 'schedulers': None}
        """
        
        raise NotImplementedError

    @torch.no_grad()
    def log_info(self) -> None:
        log_str = f'{self.config.expname} Epoch: {self.epoch}  step: {self.iteration} '
        
        for key in self.scalars_to_log.keys():
            log_str += ' {}: {:.6f}'.format(key, self.scalars_to_log[key])
            self.writer.add_scalar(key, self.scalars_to_log[key], self.iteration)
        
        print(log_str, file=self.log_file)

    def save_checkpoint(self, score: float = 0.0) -> None:
        assert self.state_dicts is not None

        self.ckpt_manager.save(
            models=self.state_dicts['models'],
            optimizers=self.state_dicts['optimizers'],
            schedulers=self.state_dicts['schedulers'],
            step=self.iteration,
            score=score
        )

    def load_checkpoint(self, load_optimizer=True, load_scheduler=True) -> int:
        iter_start = self.ckpt_manager.load(
            config=self.config,
            models=self.state_dicts['models'],
            optimizers=self.state_dicts['optimizers'] if load_optimizer else None,
            schedulers=self.state_dicts['schedulers'] if load_scheduler else None
        )

        return iter_start
