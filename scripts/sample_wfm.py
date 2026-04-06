"""
Sampling script for Wavelet Flow Matching.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import time

import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.wfm import (
    WaveletFlowMatching, apply_dwt, apply_idwt,
    MODALITY_TO_CLASS, ALL_MODALITIES
)
from tqdm.auto import tqdm


def sample_single_modality(model, wfm, batch, contr, args, device):
    """
    Sample a single modality from the unified model.

    Args:
        model: trained U-Net
        wfm: WaveletFlowMatching instance
        batch: dict with all modality tensors
        contr: target contrast name
        args: command line arguments
        device: torch device

    Returns:
        output_img: synthesized image [1, D, H, W]
        target: ground truth image [1, D, H, W]
    """
    # Get target and conditions
    target = batch[contr]
    conditions = [batch[m] for m in ALL_MODALITIES if m != contr]
    cond_1, cond_2, cond_3 = conditions
    class_idx = MODALITY_TO_CLASS[contr]

    # Apply DWT to conditions
    cond_1_dwt = apply_dwt(cond_1)
    cond_2_dwt = apply_dwt(cond_2)
    cond_3_dwt = apply_dwt(cond_3)
    cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)

    # Source: mean of conditions
    source_dwt = (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0

    # Create class labels for modality conditioning
    batch_size = target.shape[0]
    y = th.full((batch_size,), class_idx, device=device, dtype=th.long)

    # Sample
    with th.no_grad():
        output_dwt = wfm.p_sample_loop(
            model=model,
            shape=source_dwt.shape,
            cond=cond_dwt,
            source=source_dwt,
            y=y,
            num_steps=args.sampling_steps,
            device=device,
            progress=False,
            clip_denoised=args.clip_denoised,
        )

    # Convert to image domain
    output_img = apply_idwt(output_dwt)

    # Post-process
    output_img = output_img.clamp(0., 1.)
    output_img[cond_1 == 0] = 0  # Zero out non-brain regions

    return output_img, target


def main():
    args = create_argparser().parse_args()

    # Set seeds
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which modalities to sample
    if args.contr == 'all':
        target_modalities = ALL_MODALITIES
    else:
        target_modalities = [args.contr]

    logger.log("=" * 60)
    logger.log("Wavelet Flow Matching Sampling")
    logger.log("=" * 60)
    logger.log(f"Model: {args.model_path}")
    logger.log(f"Target modalities: {target_modalities}")
    logger.log(f"Sampling steps: {args.sampling_steps}")
    logger.log(f"Output directory: {args.output_dir}")
    logger.log("=" * 60)

    # Create model
    logger.log("Creating model...")
    model_args = args_to_dict(args, [
        'image_size', 'num_channels', 'num_res_blocks', 'channel_mult',
        'learn_sigma', 'class_cond', 'use_checkpoint', 'attention_resolutions',
        'num_heads', 'num_head_channels', 'num_heads_upsample',
        'use_scale_shift_norm', 'dropout', 'resblock_updown', 'use_fp16',
        'use_new_attention_order', 'dims', 'num_groups', 'in_channels',
        'out_channels', 'bottleneck_attention', 'resample_2d', 'additive_skips',
        'use_freq'
    ])

    model = create_model(**model_args)

    # Load checkpoint
    logger.log(f"Loading model from {args.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Create wfm
    wfm = WaveletFlowMatching(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
        mode='i2i'
    )

    # Load data
    logger.log("Loading validation data...")
    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='eval')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataloader = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=12,
        shuffle=False,
    )
    logger.log(f"Dataset size: {len(ds)} volumes")

    # Sampling
    logger.log(f"Starting sampling with {args.sampling_steps} steps...")

    total_time = 0
    num_samples = 0
    all_times = []

    for idx, batch in enumerate(tqdm(dataloader, desc="Sampling")):
        if args.num_samples > 0 and idx >= args.num_samples:
            break

        # Move to device
        batch = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                 for k, v in batch.items()}

        # Get subject ID
        try:
            subj = batch['subj'][0].split('/')[-2]
        except (KeyError, IndexError):
            subj = f"sample_{idx:04d}"

        # Check for missing modalities
        if batch.get('missing') != 'none' and batch.get('missing') is not None:
            logger.log(f"Skipping {subj}: missing modality {batch['missing']}")
            continue

        # Create output directory for this subject
        output_subdir = os.path.join(args.output_dir, subj)
        pathlib.Path(output_subdir).mkdir(parents=True, exist_ok=True)

        # Sample each target modality
        for contr in target_modalities:
            start_time = time.time()

            output_img, target = sample_single_modality(
                model, wfm, batch, contr, args, dist_util.dev()
            )

            sample_time = time.time() - start_time
            total_time += sample_time
            all_times.append(sample_time)
            num_samples += 1

            # Crop to original resolution
            output_img = output_img[:, :, :, :, :155]
            target = target[:, :, :, :, :155]

            # Squeeze batch and channel dimensions
            if len(output_img.shape) == 5:
                output_img = output_img.squeeze(1)
            if len(target.shape) == 5:
                target = target.squeeze(1)

            # Save synthesized image
            output_name = os.path.join(output_subdir, f'sample_{contr}.nii.gz')
            output_np = output_img[0].cpu().numpy()
            img = nib.Nifti1Image(output_np, np.eye(4))
            nib.save(img=img, filename=output_name)

            # Save target for comparison
            target_name = os.path.join(output_subdir, f'target_{contr}.nii.gz')
            target_np = target[0].cpu().numpy()
            img = nib.Nifti1Image(target_np, np.eye(4))
            nib.save(img=img, filename=target_name)

        if (idx + 1) % 10 == 0:
            logger.log(f"Processed {idx + 1} subjects, avg time: {total_time/num_samples:.2f}s")

    # Summary
    logger.log("=" * 60)
    logger.log("SAMPLING COMPLETE")
    logger.log("=" * 60)
    logger.log(f"Total samples: {num_samples}")
    logger.log(f"Modalities per subject: {len(target_modalities)}")
    logger.log(f"Sampling steps: {args.sampling_steps}")
    logger.log(f"Average time per sample: {total_time/num_samples:.2f}s")
    logger.log(f"Total time: {total_time:.1f}s")

    if all_times:
        logger.log(f"Min time: {min(all_times):.2f}s")
        logger.log(f"Max time: {max(all_times):.2f}s")
        logger.log(f"Std time: {np.std(all_times):.2f}s")

    logger.log(f"Results saved to: {args.output_dir}")
    logger.log("=" * 60)

    # Save timing summary
    timing_file = os.path.join(args.output_dir, 'timing.txt')
    with open(timing_file, 'w') as f:
        f.write(f"Sampling Steps: {args.sampling_steps}\n")
        f.write(f"Modalities: {target_modalities}\n")
        f.write(f"Total Samples: {num_samples}\n")
        f.write(f"Average Time (s): {total_time/num_samples:.2f}\n")
        f.write(f"Total Time (s): {total_time:.1f}\n")
        if all_times:
            f.write(f"Min Time (s): {min(all_times):.2f}\n")
            f.write(f"Max Time (s): {max(all_times):.2f}\n")
            f.write(f"Std Time (s): {np.std(all_times):.2f}\n")


def create_argparser():
    defaults = dict(
        # Sampling settings
        seed=42,
        data_dir="",
        model_path="",
        output_dir="./results_wfm/",
        devices=[0],
        num_samples=-1,  # -1 for all samples
        sampling_steps=50,  # Variable: 10, 50, 100, 200
        contr='all',  # 'all' for all modalities, or specific: 't1n', 't1c', 't2w', 't2f'
        clip_denoised=True,
        dataset='brats',

        # Bridge settings
        sigma_max=0.5,

        # Model settings (same as training)
        dims=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=False,
        use_scale_shift_norm=False,
        attention_resolutions="",
        channel_mult="1,2,2,4,4",
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        num_groups=32,
        in_channels=32,
        out_channels=8,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
        predict_xstart=True,
        image_size=224,
        class_cond=True,  # Enable class conditioning for unified model
        dropout=0.0,
        use_checkpoint=False,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        num_head_channels=-1,
        num_heads_upsample=-1,
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
