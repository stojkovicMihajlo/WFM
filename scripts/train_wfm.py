"""
Training script for Wavelet Flow Matching.
"""

import argparse
import numpy as np
import os
import random
import sys
import time

import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

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
    WaveletFlowMatching, apply_idwt,
    MODALITY_TO_CLASS, ALL_MODALITIES
)
from guided_diffusion.resample import UniformSampler
from guided_diffusion.train_util import visualize


class BridgeTrainLoop:
    """Training loop for Wavelet Flow Matching."""

    def __init__(
        self,
        model,
        wfm,
        data,
        batch_size,
        lr,
        weight_decay,
        log_interval,
        save_interval,
        resume_step,
        max_iterations,
        contr,
        summary_writer=None,
        dataset='brats',
    ):
        self.model = model
        self.wfm = wfm
        self.datal = data
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_step = resume_step
        self.max_iterations = max_iterations
        self.contr = contr  # 'all' for unified training, or specific modality
        self.summary_writer = summary_writer
        self.dataset = dataset

        self.step = 1
        self.global_batch = self.batch_size * dist.get_world_size()

        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Load optimizer state if resuming
        if self.resume_step > 0:
            self._load_optimizer_state()

    def _load_optimizer_state(self):
        """Load optimizer state from checkpoint."""
        opt_checkpoint = os.path.join(
            logger.get_dir(),
            'checkpoints',
            f"opt_wfm_{self.resume_step:06d}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
        else:
            logger.log(f"No optimizer checkpoint found at {opt_checkpoint}")

    def run_loop(self):
        """Main training loop."""
        t_start = time.time()

        while self.step + self.resume_step < self.max_iterations:
            t_iter = time.time()

            # Get batch
            try:
                batch = next(self.iterdatal)
            except StopIteration:
                self.iterdatal = iter(self.datal)
                batch = next(self.iterdatal)

            # Move to device
            batch = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                     for k, v in batch.items()}

            t_load = time.time() - t_iter

            # Training step
            t_fwd = time.time()
            loss, sample, sample_idwt = self.run_step(batch)
            t_fwd = time.time() - t_fwd

            # Logging
            if self.summary_writer is not None:
                global_step = self.step + self.resume_step
                self.summary_writer.add_scalar('time/load', t_load, global_step=global_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=global_step)
                self.summary_writer.add_scalar('loss/MSE', loss.item(), global_step=global_step)

                # Log per-channel losses
                names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                for ch, name in enumerate(names):
                    if hasattr(self, '_last_channel_losses') and len(self._last_channel_losses) > ch:
                        self.summary_writer.add_scalar(
                            f'loss/mse_wav_{name.lower()}',
                            self._last_channel_losses[ch].item(),
                            global_step=global_step
                        )

            if self.step % 200 == 0 and self.summary_writer is not None:
                # Log sample images
                global_step = self.step + self.resume_step
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2]
                self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                              global_step=global_step)

                # Log wavelet channels
                names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                image_size = sample.size()[2]
                for ch, name in enumerate(names):
                    midplane = sample[0, ch, :, :, image_size // 2]
                    self.summary_writer.add_image(f'sample/{name}', midplane.unsqueeze(0),
                                                  global_step=global_step)

            if self.step % self.log_interval == 0:
                elapsed = time.time() - t_start
                logger.log(f"step {self.step + self.resume_step}: loss={loss.item():.6f}, "
                           f"elapsed={elapsed:.1f}s, iter/s={self.step/elapsed:.2f}")
                logger.dumpkvs()

            # Save checkpoint
            if self.step % self.save_interval == 0:
                self.save()

            self.step += 1

        # Final save
        if (self.step - 1) % self.save_interval != 0:
            self.save()

        logger.log("Training complete!")

    def _select_modality(self):
        """Select target modality for this training step."""
        if self.contr == 'all':
            # Unified training: randomly select a modality each step
            return random.choice(ALL_MODALITIES)
        else:
            # Fixed modality training (backwards compatible)
            return self.contr

    def run_step(self, batch):
        """Single training step."""
        # Zero gradients
        self.opt.zero_grad()

        # Select modality for this step
        contr = self._select_modality()

        # Sample timesteps uniformly
        t = th.randint(0, self.wfm.num_timesteps, (self.batch_size,),
                       device=dist_util.dev())

        # Compute loss
        losses, model_output, model_output_idwt = self.wfm.training_losses(
            model=self.model,
            x_start=batch,
            t=t,
            mode='i2i',
            contr=contr,
        )

        # Store per-channel losses for logging
        self._last_channel_losses = losses["mse_wav"]
        self._last_modality = contr

        # Total loss (mean over channels)
        loss = losses["mse_wav"].mean()

        # Check for NaN
        if not th.isfinite(loss):
            logger.log(f"Warning: non-finite loss {loss}", level=logger.WARN)

        # Backward
        loss.backward()

        # Gradient clipping (optional but recommended for stability)
        th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.opt.step()

        # Log step
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("loss", loss.item())
        if self.contr == 'all':
            logger.logkv("modality", contr)

        return loss.detach(), model_output, model_output_idwt

    def save(self):
        """Save model and optimizer checkpoints."""
        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            step = self.step + self.resume_step

            # Save model - use 'unified' for all-modality training
            model_name = 'unified' if self.contr == 'all' else self.contr
            model_path = os.path.join(
                checkpoint_dir,
                f"wfm_{model_name}_{step:06d}.pt"
            )
            logger.log(f"Saving model to {model_path}")
            th.save(self.model.state_dict(), model_path)

            # Save optimizer
            opt_path = os.path.join(
                checkpoint_dir,
                f"opt_wfm_{step:06d}.pt"
            )
            th.save(self.opt.state_dict(), opt_path)


def main():
    args = create_argparser().parse_args()

    # Set seeds
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup distributed
    dist_util.setup_dist(devices=args.devices)

    # Setup logging
    summary_writer = None
    if args.use_tensorboard:
        logdir = args.tensorboard_path if args.tensorboard_path else None
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    # Determine training mode
    is_unified = args.contr == 'all'
    mode_str = "Unified (all modalities)" if is_unified else f"Single modality ({args.contr})"

    logger.log("=" * 60)
    logger.log("Wavelet Flow Matching Training")
    logger.log("=" * 60)
    logger.log(f"Training mode: {mode_str}")
    logger.log(f"Class conditioning: {args.class_cond}")
    logger.log(f"Sigma max: {args.sigma_max}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Learning rate: {args.lr}")
    logger.log(f"Max iterations: {args.max_iterations}")
    if args.resume_step > 0:
        logger.log(f"Resume step: {args.resume_step}")
    logger.log("=" * 60)

    # Create model (same architecture as cWDM)
    logger.log("Creating model...")

    # Parse arguments for model creation
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
    model.to(dist_util.dev())

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model parameters: {num_params:,}")

    # Load checkpoint if resuming
    if args.resume_checkpoint:
        logger.log(f"Loading model from {args.resume_checkpoint}")
        model.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint, map_location=dist_util.dev())
        )

    # Create wfm diffusion
    wfm = WaveletFlowMatching(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
        mode='i2i'
    )

    # Load data
    logger.log("Loading data...")
    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='train')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataloader = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    logger.log(f"Dataset size: {len(ds)} volumes")

    # Create training loop
    logger.log("Starting training...")
    train_loop = BridgeTrainLoop(
        model=model,
        wfm=wfm,
        data=dataloader,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_step=args.resume_step,
        max_iterations=args.max_iterations,
        contr=args.contr,
        summary_writer=summary_writer,
        dataset=args.dataset,
    )

    train_loop.run_loop()


def create_argparser():
    defaults = dict(
        # Training settings
        seed=42,
        data_dir="",
        lr=1e-5,
        weight_decay=0.0,
        batch_size=4,
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',
        resume_step=0,
        max_iterations=200000,
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        num_workers=8,
        dataset='brats',

        # Target contrast: 'all' for unified model, or specific modality
        contr='all',  # 'all' (unified), 't1n', 't1c', 't2w', 't2f'

        # Bridge-specific settings
        sigma_max=0.5,  # Maximum wfm noise level

        # Model settings (same as cWDM)
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
        in_channels=32,  # 8 noisy + 24 condition
        out_channels=8,  # 8 wavelet channels
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
