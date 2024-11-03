# GGRt: Towards Pose-free Generalizable 3D Gaussian Splatting in Real-time
[[Project Page](https://3d-aigc.github.io/GGRt/) | [arXiv](https://arxiv.org/pdf/2403.10147.pdf)] | [Model](https://drive.google.com/drive/folders/1Y-0YeTkoUQHGnuA_IZrFbcy8iwH9FE4J?usp=drive_link)
# Installation
```bash
git clone https://github.com/dcharatan/diff-gaussian-rasterization-modified
pip install -e ./diff-gaussian-rasterization-modified
```

## Data Preparation
For `LLFF` dataset, please follow the [ibrnet](https://github.com/googleinterns/IBRNet). For `Waymo` dataset, please download the data from [EmerNeRF](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md).

The data structure is as follows:
```bash
data
├── ibrnet                
│   ├── train
│   │   ├── real_iconic_noface
│   │   │   ├── airplants
│   │   │   ├── ...
│   │   ├── ibrnet_collected_1
│   │   │   ├── ...
│   │   ├── ibrnet_collected_2
│   │   │   ├── ...
│   ├── train
├── nerf_llff_data
│   ├── fern
│   ├── room
│   ├── ...
├── waymo
│   ├── training
│   │   ├── 139
│   │   ├── 140
│   │   ├── ...   
│   ├── testing
│   │   ├── 003
│   │   ├── 019
│   │   ├── ...   
```
### On LLFF Dataset
we provide `launch.json` formatted for debugging.
#### Generalizable Training 
```json
{
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "${workspaceFolder}"
    },
    "name": "generalize:ggrt-llff",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/train_ggrt_stable.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "++rootdir=data/ibrnet/train",
        "+ckpt_path=model_zoo/generalized_llff_best.pth",
        "++train_dataset=llff",
        "++eval_dataset=llff",
        "++num_source_views=4",
        "++expname=generalizable_llff",
        "++use_depth_loss=False",
        "++use_pred_pose=True",
        "++render_video=False",
        "++crop_size=2",
    ]
},
```
#### Finetune on Sepcific Scenes
```json
{
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "${workspaceFolder}"
    },
    "name": "finetune:ggrt-waymo",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/finetune_ggrt_stable.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "++rootdir=data/ibrnet/train",
        "+ckpt_path=model_zoo/generalized_llff_best.pth",
        "++train_dataset=llff_test",
        "++eval_dataset=llff_test",
        "++eval_scenes=[fern]",
        "++train_scenes=[fern]",
        "++num_source_views=4",
        "++expname=ft_llff_fern",
        "++use_depth_loss=False",
        "++use_pred_pose=True",
        "++render_video=False",
        "++crop_size=2",
    ]
},
```
#### Evaluation
```json
{
    "env": {
        "CUDA_VISIBLE_DEVICES": "7",
        "PYTHONPATH": "${workspaceFolder}"
    },
    "name": "test:ggrt-llff",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/eval/eval_ggrt.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "++rootdir=data/ibrnet/eval",
        "+ckpt_path=model_zoo/generalized_llff_best.pth", 
        "++train_dataset=llff_test",
        "++train_scenes=[fern]",
        "++eval_dataset=llff_test",
        "++eval_scenes=[fern]",
        "++num_source_views=5",
        "++render_video=False",
        "++expname=generalizable_llff_fern",
    ]
}
```

### On Waymo Dataset
#### Generalizable Training 
```json
{
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "${workspaceFolder}"
    },
    "name": "generalize:ggrt-waymo",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/train_ggrt_stable.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "++rootdir=data/ibrnet/train",
        "+ckpt_path=model_zoo/generalized_waymo_best.pth",
        "++train_dataset=waymo",
        "++eval_dataset=waymo",
        "++eval_scenes=[019]",
        "++num_source_views=4",
        "++expname=ft_waymo_019",
        "++use_depth_loss=False",
        "++use_pred_pose=True",
        "++render_video=False",
        "++crop_size=2",
    ]
},
```
#### Finetune on Sepcific Scenes
```json
{
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "${workspaceFolder}"
    },
    "name": "finetune:ggrt-waymo",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/finetune_ggrt_stable.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "++rootdir=data/ibrnet/train",
        "+ckpt_path=model_zoo/generalized_waymo_best.pth",
        "++train_dataset=waymo",
        "++eval_dataset=waymo",
        "++eval_scenes=[019]",
        "++train_scenes=[019]",
        "++num_source_views=4",
        "++expname=ft_waymo_019",
        "++use_depth_loss=False",
        "++use_pred_pose=True",
        "++render_video=False",
        "++crop_size=2",
    ]
},
```
#### Evaluation
```json
{
    "env": {
        "CUDA_VISIBLE_DEVICES": "7",
        "PYTHONPATH": "${workspaceFolder}"
    },
    "name": "test:ggrt-waymo",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/eval/eval_ggrt.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "++rootdir=data/ibrnet/eval",
        "+ckpt_path=model_zoo/generalized_waymo_best.pth", 
        "++train_dataset=waymo",
        "++train_scenes=[019]",
        "++eval_dataset=waymo",
        "++eval_scenes=['019']",
        "++num_source_views=5",
        "++render_video=False",
        "++expname=opensource",
        "++dataset_root_eval=data/waymo/testing"  // 测试集路径
    ]
}
```