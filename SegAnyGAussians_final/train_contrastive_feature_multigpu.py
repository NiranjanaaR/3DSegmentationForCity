#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_renderer import render_contrastive_feature
import sys
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np


import torch
from torch import nn
import pytorch3d.ops


import time
##niranjana added 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
#ends

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from sklearn.preprocessing import QuantileTransformer
# Borrowed from GARField but modified
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can change the port if needed
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    def quantile_transformer_func(scales):
        # This function acts as a wrapper for QuantileTransformer.
        # QuantileTransformer expects a numpy array, while we have a torch tensor.
        scales = scales.reshape(-1,1)
        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device)

    return quantile_transformer_func

from torch.cuda.amp import autocast, GradScaler

def training(dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from, rank, world_size):
    print("RFN weight:", opt.rfn)
    print("Smooth K:", opt.smooth_K)
    print("Scale aware dim:", opt.scale_aware_dim)
    torch.cuda.set_device(rank)
    assert opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0

    dataset.need_features = True
    dataset.need_masks = True

    gaussians = GaussianModel(dataset.sh_degree)
    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    sample_rate = 0.2 if 'Replica' in dataset.source_path else 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)
    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False)

    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, 32, bias=True),
        torch.nn.Sigmoid()
    ).cuda().train()

    param_group = {'params': scale_gate.parameters(), 'lr': opt.feature_lr, 'name': 'f'}
    feature_gaussians.optimizer.add_param_group(param_group)

    smooth_weights = None

    del gaussians
    torch.cuda.empty_cache()

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    print("Preparing Quantile Transform...")
    all_scales = []
    for cam in scene.getTrainCameras():
        all_scales.append(cam.mask_scales)
    all_scales = torch.cat(all_scales)
    upper_bound_scale = all_scales.max().item()

    scale_aware_dim = opt.scale_aware_dim
    if scale_aware_dim <= 0 or scale_aware_dim >= 32:
        print("Using adaptive scale gate.")
        q_trans = get_quantile_func(all_scales, "uniform")
    else:
        q_trans = get_quantile_func(all_scales, "uniform")
        fixed_scale_gate = torch.tensor(
            [[1 for j in range(32 - scale_aware_dim + i)] + [0 for k in range(scale_aware_dim - i)] for i in range(scale_aware_dim+1)]
        ).cuda()

    scaler = GradScaler()

    viewpoint_stack = None

    for iter_idx in progress_bar:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        with torch.no_grad():
            sam_masks = viewpoint_cam.original_masks.cuda().float()
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

            mask_scales = viewpoint_cam.mask_scales.cuda()
            mask_scales, sort_indices = torch.sort(mask_scales, descending=True)
            sam_masks = sam_masks[sort_indices, :, :]

            num_sampled_scales = 4
            sampled_scale_index = torch.randperm(len(mask_scales))[:num_sampled_scales]
            tmp = torch.zeros(num_sampled_scales+2)
            tmp[1:len(sampled_scale_index)+1] = sampled_scale_index
            tmp[-1] = len(mask_scales) - 1
            tmp[0] = -1
            sampled_scale_index = tmp.long()
            sampled_scales = mask_scales[sampled_scale_index]

            second_big_scale = mask_scales[mask_scales < upper_bound_scale].max()

            ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else opt.num_sampled_rays / (sam_masks.shape[-1] * sam_masks.shape[-2])

            sampled_ray = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda() < ray_sample_rate
            non_mask_region = sam_masks.sum(dim=0) == 0
            sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region)

            per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:,None,None]
            per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim=0) / (sam_masks.sum(dim=0) + 1e-9)
            per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray]

            pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1)
            ptp_max_size = pixel_to_pixel_mask_size.max()
            pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
            per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
            per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min()) * 9. + 1.

            sam_masks_sampled_ray = sam_masks[:, sampled_ray]

            gt_corrs = []
            sampled_scales[0] = upper_bound_scale + upper_bound_scale * torch.rand(1)[0]
            for idx, si in enumerate(sampled_scale_index):
                upper_bound = sampled_scales[idx] >= upper_bound_scale
                if si != len(mask_scales) - 1 and not upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - mask_scales[si+1]) * torch.rand(1)[0]
                elif upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - second_big_scale) * torch.rand(1)[0]
                else:
                    sampled_scales[idx] -= sampled_scales[idx] * torch.rand(1)[0]

                if not upper_bound:
                    gt_vec = torch.zeros_like(sam_masks_sampled_ray)
                    gt_vec[:si+1,:] = sam_masks_sampled_ray[:si+1,:]
                    for j in range(si, -1, -1):
                        gt_vec[j,:] = torch.logical_and(~gt_vec[j+1:,:].any(dim=0), gt_vec[j,:])
                    gt_vec[si+1:,:] = sam_masks_sampled_ray[si+1:,:]
                else:
                    gt_vec = sam_masks_sampled_ray

                gt_corr = torch.einsum('nh,nj->hj', gt_vec, gt_vec)
                gt_corr[gt_corr != 0] = 1
                gt_corrs.append(gt_corr)

            gt_corrs = torch.stack(gt_corrs, dim=0)
            sampled_scales = q_trans(sampled_scales).squeeze()

        # 🎯 Start autocast
        with autocast():
            render_pkg_feat = feature_gaussians(
                viewpoint_cam,
                pipe,
                background,
                norm_point_features=True,
                smooth_type='traditional',
                smooth_weights=torch.softmax(smooth_weights, dim=-1) if smooth_weights is not None else None,
                smooth_K=opt.smooth_K
            )
            rendered_features = render_pkg_feat["render"]
            rendered_feature_norm = rendered_features.norm(dim=0, p=2).mean()
            rendered_feature_norm_reg = (1-rendered_feature_norm)**2

            rendered_features = torch.nn.functional.interpolate(
                rendered_features.unsqueeze(0),
                viewpoint_cam.original_masks.shape[-2:],
                mode='bilinear'
            ).squeeze(0)

            feature_with_scale = rendered_features.unsqueeze(0).repeat([sampled_scales.shape[0],1,1,1])

            gates = scale_gate(sampled_scales.unsqueeze(-1))

            feature_with_scale = feature_with_scale * gates.unsqueeze(-1).unsqueeze(-1)

            sampled_feature_with_scale = feature_with_scale[:,:,sampled_ray]
            scale_conditioned_features_sam = sampled_feature_with_scale.permute([0,2,1])

            scale_conditioned_features_sam = torch.nn.functional.normalize(scale_conditioned_features_sam, dim=-1, p=2)
            corr = torch.einsum('nhc,njc->nhj', scale_conditioned_features_sam, scale_conditioned_features_sam)

            diag_mask = torch.eye(corr.shape[1], dtype=bool, device=corr.device)

            sum_0 = gt_corrs.sum(dim=0)
            consistent_negative = sum_0 == 0
            consistent_positive = sum_0 == len(gt_corrs)
            inconsistent = ~(consistent_negative | consistent_positive)
            inconsistent_num = inconsistent.count_nonzero()
            sampled_num = inconsistent_num / 2
            rand_num = torch.rand_like(sum_0)

            sampled_positive = consistent_positive & (rand_num < sampled_num / consistent_positive.count_nonzero())
            sampled_negative = consistent_negative & (rand_num < sampled_num / consistent_negative.count_nonzero())

            sampled_mask_positive = ((sampled_positive | torch.any((corr < 0.75) & (gt_corrs == 1), dim=0)) | inconsistent) & ~diag_mask
            sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=0)

            sampled_mask_negative = ((sampled_negative | torch.any((corr > 0.5) & (gt_corrs == 0), dim=0)) | inconsistent) & ~diag_mask
            sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=0)

            per_pixel_weight = per_pixel_weight.unsqueeze(0)

            loss = (-per_pixel_weight[:, sampled_mask_positive] * gt_corrs[:, sampled_mask_positive] * corr[:, sampled_mask_positive]).mean() \
                 + (per_pixel_weight[:, sampled_mask_negative] * (1 - gt_corrs[:, sampled_mask_negative]) * torch.relu(corr[:, sampled_mask_negative])).mean() \
                 + opt.rfn * rendered_feature_norm_reg

        scaler.scale(loss).backward()
        scaler.step(feature_gaussians.optimizer)
        scaler.update()
        feature_gaussians.optimizer.zero_grad(set_to_none=True)

        if iter_idx % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Loss": f"{loss.item():.{3}f}"
            })

    scene.save_feature(opt.iterations, target='contrastive_feature', smooth_weights=torch.softmax(smooth_weights, dim=-1) if smooth_weights is not None else None, smooth_type='traditional', smooth_K=opt.smooth_K)
    if rank == 0:
        print('Saving model')
        torch.save(scale_gate.state_dict(), os.path.join(scene.model_path, f"point_cloud/iteration_{opt.iterations}/scale_gate.pt"))


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
#added for ddp niranjana
def ddp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Directly extract from pre-parsed args
    model_params = args.model_params
    optim_params = args.optim_params
    pipe_params = args.pipe_params
    

    training(model_params, optim_params, pipe_params,
             args.iteration, args.save_iterations, args.checkpoint_iterations,
             args.debug_from, rank, world_size)

    cleanup()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--target', default='contrastive_feature', const='contrastive_feature', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature'])
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--model_path', type=str, default=None)

    # Load main args
    args = get_combined_args(parser, target_cfg_file='cfg_args')
    args.save_iterations.append(args.iterations)
   
    # Handle model_path separately
    temp_model_path = args.model_path
    parser_model = ArgumentParser()
    args.model_params = ModelParams(parser_model).extract(args)
    args.model_params.model_path = temp_model_path

    # Optimization and pipeline parameters
    parser_opt = ArgumentParser()
    args.optim_params = OptimizationParams(parser_opt).extract(args)

    # ✅ Patch missing attributes if needed (fail-safe, should be unnecessary if all works fine)
    if not hasattr(args.optim_params, 'position_lr_init'):
        args.optim_params.position_lr_init = 0.00016
    if not hasattr(args.optim_params, 'position_lr_final'):
        args.optim_params.position_lr_final = 0.0000016
    if not hasattr(args.optim_params, 'percent_dense'):
        args.optim_params.percent_dense = 0.01
    if not hasattr(args.optim_params, 'position_lr_delay_mult'):
        args.optim_params.position_lr_delay_mult = 0.01
    if not hasattr(args.optim_params, 'position_lr_max_steps'):
        args.optim_params.position_lr_max_steps = 30_000
    if not hasattr(args.optim_params, 'feature_lr'):
        args.optim_params.feature_lr = 0.01
    if not hasattr(args.optim_params, 'lambda_dssim'):
        args.optim_params.lambda_dssim = 0.2

    args.pipe_params = PipelineParams(parser).extract(args)

    # ✅ Sanity check
    print("position_lr_init:", args.optim_params.position_lr_init)
    print("position_lr_final:", args.optim_params.position_lr_final)
    print("percent_dense:", args.optim_params.percent_dense)
  
    # Fire up training
    print("Optimizing")
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    world_size = torch.cuda.device_count()
    mp.spawn(ddp_main, args=(world_size, args), nprocs=world_size, join=True)

    print("\nTraining complete.")
