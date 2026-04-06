"""
Wavelet Flow Matching for 3D multi-modal MRI synthesis.

Forward process: x_t = (1-t)*x_source + t*x_target + σ(t)*noise
Model predicts: velocity v = x_target - x_source

Loss (Eq. 3): L = E[||f_θ(x_t, c, t, y) - (x_target - x_source)||²]

Unified model with class conditioning for all 4 BraTS modalities.
"""

import torch
import torch as th
import numpy as np
from .nn import mean_flat

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dwt = DWT_3D('haar')
idwt = IDWT_3D('haar')

# Modality to class index mapping for unified model
MODALITY_TO_CLASS = {'t1n': 0, 't1c': 1, 't2w': 2, 't2f': 3}
CLASS_TO_MODALITY = {0: 't1n', 1: 't1c', 2: 't2w', 3: 't2f'}
ALL_MODALITIES = ['t1n', 't1c', 't2w', 't2f']


def apply_dwt(img):
    """Apply 3D DWT and concatenate subbands along channel dimension."""
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(img)
    return th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)


def apply_idwt(x):
    """Apply 3D IDWT to reconstruct image from wavelet subbands."""
    B, _, H, W, D = x.size()
    return idwt(
        x[:, 0, :, :, :].view(B, 1, H, W, D) * 3.,
        x[:, 1, :, :, :].view(B, 1, H, W, D),
        x[:, 2, :, :, :].view(B, 1, H, W, D),
        x[:, 3, :, :, :].view(B, 1, H, W, D),
        x[:, 4, :, :, :].view(B, 1, H, W, D),
        x[:, 5, :, :, :].view(B, 1, H, W, D),
        x[:, 6, :, :, :].view(B, 1, H, W, D),
        x[:, 7, :, :, :].view(B, 1, H, W, D)
    )


class WaveletFlowMatching:
    """
    Wavelet Flow Matching for paired image-to-image translation.

    Forward process: x_t = (1-t)*x_source + t*x_target + σ(t)*noise
    Model predicts:  velocity v = x_target - x_source
    """

    def __init__(
        self,
        num_timesteps=1000,
        sigma_max=0.5,
        mode='i2i',
    ):
        """
        Initialize Wavelet Flow Matching.

        Args:
            num_timesteps: Number of discrete timesteps (for compatibility with U-Net)
            sigma_max: Maximum noise level (peaks at t=0.5)
            mode: Operating mode, 'i2i' for image-to-image translation
        """
        self.num_timesteps = num_timesteps
        self.sigma_max = sigma_max
        self.mode = mode

        # Precompute sigma schedule (peaks at t=0.5)
        t = np.linspace(0, 1, num_timesteps)
        self.sigmas = sigma_max * np.sqrt(t * (1 - t))

    def get_sigma(self, t):
        """
        Flow matching noise schedule: σ(t) = σ_max * sqrt(t * (1-t))
        Peaks at t=0.5, zero at endpoints.

        This schedule ensures:
        - At t=0: x_t = x_source (no noise, pure source)
        - At t=1: x_t = x_target (no noise, pure target)
        - At t=0.5: Maximum uncertainty/noise

        Args:
            t: timestep in [0, 1], shape [B] or scalar
        Returns:
            sigma: noise level, same shape as t
        """
        return self.sigma_max * torch.sqrt(t * (1 - t) + 1e-8)

    def q_sample(self, x_source, x_target, t, noise=None):
        """
        Forward process: interpolate source -> target with noise.

        x_t = (1 - t) * x_source + t * x_target + σ(t) * noise

        Args:
            x_source: wavelet coefficients of source [B, 8, D, H, W]
            x_target: wavelet coefficients of target [B, 8, D, H, W]
            t: timestep in [0, 1], shape [B]
            noise: optional pre-sampled noise

        Returns:
            x_t: noisy interpolation at time t
        """
        if noise is None:
            noise = torch.randn_like(x_target)

        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1, 1]
        t_broadcast = t.view(-1, 1, 1, 1, 1)

        # Linear interpolation between source and target
        x_t = (1 - t_broadcast) * x_source + t_broadcast * x_target

        # Add noise scaled by sigma(t)
        sigma_t = self.get_sigma(t).view(-1, 1, 1, 1, 1)
        x_t = x_t + sigma_t * noise

        return x_t

    def _scale_timesteps(self, t):
        """
        Scale continuous timesteps [0,1] to discrete range [0, num_timesteps].
        This maintains compatibility with the pretrained U-Net timestep embedding.
        """
        return t * self.num_timesteps

    def _get_target_and_conditions(self, x_start, contr):
        """
        Get target and conditioning modalities based on target contrast.

        Args:
            x_start: dict with 't1n', 't1c', 't2w', 't2f' tensors
            contr: target contrast name or class index

        Returns:
            target: target modality tensor
            conditions: list of 3 conditioning modality tensors
            class_idx: integer class index for the target modality
        """
        # Handle both string and integer inputs
        if isinstance(contr, int):
            class_idx = contr
            contr = CLASS_TO_MODALITY[contr]
        else:
            class_idx = MODALITY_TO_CLASS[contr]

        # Get target modality
        target = x_start[contr]

        # Get conditioning modalities (all others)
        conditions = [x_start[m] for m in ALL_MODALITIES if m != contr]

        return target, conditions, class_idx

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None,
                        mode='i2i', contr='t1n'):
        """
        Compute flow matching training loss (Eq. 3 from paper).

        Loss: L = E[||f_θ(x_t, c, t, y) - (x_target - x_source)||²]

        Model learns velocity field v = x_target - x_source.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get target, conditions, and class index
        target, conditions, class_idx = self._get_target_and_conditions(x_start, contr)
        cond_1, cond_2, cond_3 = conditions

        # Apply DWT to conditions and concatenate [B, 24, D/2, H/2, W/2]
        cond_1_dwt = apply_dwt(cond_1)
        cond_2_dwt = apply_dwt(cond_2)
        cond_3_dwt = apply_dwt(cond_3)
        cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)

        # Apply DWT to target [B, 8, D/2, H/2, W/2]
        target_dwt = apply_dwt(target)

        # Create source from conditions (mean of 3 modalities in wavelet space)
        # This provides a reasonable starting point (informed prior)
        source_dwt = (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0

        # Convert discrete timestep to continuous t ∈ [0, 1]
        t_continuous = t.float() / self.num_timesteps

        # Sample noise and apply DWT
        if noise is None:
            noise = th.randn_like(target)
        noise_dwt = apply_dwt(noise)

        # Forward: interpolate source -> target with noise
        x_t = self.q_sample(source_dwt, target_dwt, t_continuous, noise=noise_dwt)

        # Concatenate condition for conditional generation [B, 32, D/2, H/2, W/2]
        x_t_cond = th.cat([x_t, cond_dwt], dim=1)

        # Create class labels for modality conditioning [B]
        batch_size = target.shape[0]
        y = th.full((batch_size,), class_idx, device=target.device, dtype=th.long)

        # Model predicts velocity v = x_target - x_source
        t_scaled = self._scale_timesteps(t_continuous)
        model_output = model(x_t_cond, t_scaled, y=y, **model_kwargs)

        # Velocity loss (Eq. 3): ||f_θ(x_t) - (x_target - x_source)||²
        velocity = target_dwt - source_dwt
        mse_per_channel = mean_flat((velocity - model_output) ** 2)
        loss = th.mean(mse_per_channel, dim=0)

        # For visualization: predicted target = source + velocity
        pred_target = source_dwt + model_output
        model_output_idwt = apply_idwt(pred_target)

        terms = {"mse_wav": loss}
        return terms, model_output, model_output_idwt

    def p_sample_step_euler(self, model, x_t, t_curr, t_next, cond, source, y=None,
                            model_kwargs=None, deterministic=True):
        """Euler integration step. Model predicts velocity directly."""
        if model_kwargs is None:
            model_kwargs = {}

        B = x_t.shape[0]
        device = x_t.device

        t_model = torch.full((B,), t_curr * self.num_timesteps, device=device)
        x_t_cond = torch.cat([x_t, cond], dim=1)

        with torch.no_grad():
            velocity = model(x_t_cond, t_model, y=y, **model_kwargs)

        dt = t_next - t_curr
        x_next = x_t + dt * velocity

        return x_next

    def p_sample_step_heun(self, model, x_t, t_curr, t_next, cond, source, y=None,
                           model_kwargs=None):
        """Heun's method (RK2). Model predicts velocity directly."""
        if model_kwargs is None:
            model_kwargs = {}

        B = x_t.shape[0]
        device = x_t.device
        dt = t_next - t_curr

        t_model_curr = torch.full((B,), t_curr * self.num_timesteps, device=device)
        x_t_cond = torch.cat([x_t, cond], dim=1)

        with torch.no_grad():
            v1 = model(x_t_cond, t_model_curr, y=y, **model_kwargs)

        x_pred = x_t + dt * v1

        t_model_next = torch.full((B,), t_next * self.num_timesteps, device=device)
        x_pred_cond = torch.cat([x_pred, cond], dim=1)

        with torch.no_grad():
            v2 = model(x_pred_cond, t_model_next, y=y, **model_kwargs)

        x_next = x_t + dt * 0.5 * (v1 + v2)

        return x_next

    def p_sample_step(self, model, x_t, t_curr, t_next, cond, source, y=None,
                      model_kwargs=None, clip_denoised=True, solver='euler'):
        """Single sampling step using specified solver."""
        if solver == 'heun':
            return self.p_sample_step_heun(model, x_t, t_curr, t_next, cond, source, y, model_kwargs)
        else:
            return self.p_sample_step_euler(model, x_t, t_curr, t_next, cond, source, y, model_kwargs)

    def direct_prediction(self, model, cond, source, y=None, device=None, model_kwargs=None):
        """Single-step prediction: source + velocity."""
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device

        B = source.shape[0]
        t_model = torch.full((B,), self.num_timesteps, device=device, dtype=torch.float32)
        x_cond = torch.cat([source, cond], dim=1)

        with torch.no_grad():
            velocity = model(x_cond, t_model, y=y, **model_kwargs)

        return source + velocity

    def p_sample_loop(
        self,
        model,
        shape,
        cond,
        source,
        y=None,
        num_steps=50,
        device=None,
        progress=True,
        clip_denoised=True,
        model_kwargs=None,
        solver='euler',
    ):
        """Generate samples by integrating the learned velocity field."""
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device

        # Direct prediction mode - single forward pass
        if solver == 'direct' or num_steps == 1:
            output = self.direct_prediction(model, cond, source, y, device, model_kwargs)
            if clip_denoised:
                output = self._clip_wavelets(output)
            return output

        # Start from source
        x_t = source.clone()

        # Timesteps from 0 to 1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

        if progress:
            from tqdm.auto import tqdm
            solver_name = solver.upper()
            iterator = tqdm(range(num_steps), desc=f"WFM {solver_name} ({num_steps} steps)")
        else:
            iterator = range(num_steps)

        for i in iterator:
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()

            x_t = self.p_sample_step(
                model=model,
                x_t=x_t,
                t_curr=t_curr,
                t_next=t_next,
                cond=cond,
                source=source,
                y=y,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
                solver=solver,
            )

        # Optional: clip in image domain for final output
        if clip_denoised:
            x_t = self._clip_wavelets(x_t)

        return x_t

    def _clip_wavelets(self, x):
        """Clip wavelets via image domain clamping to [0,1]."""
        x_idwt = apply_idwt(x)
        x_idwt_clamp = x_idwt.clamp(0., 1.)
        return apply_dwt(x_idwt_clamp)

    def sample_for_evaluation(
        self,
        model,
        batch,
        contr='t1n',
        num_steps=50,
        device=None,
        progress=True,
        clip_denoised=True,
    ):
        """Sample a modality from batch dict, returns (synthesized, ground_truth)."""
        if device is None:
            device = next(model.parameters()).device

        # Get target, conditions, and class index using helper
        target, conditions, class_idx = self._get_target_and_conditions(batch, contr)
        cond_1, cond_2, cond_3 = conditions

        # Apply DWT to conditions
        cond_1_dwt = apply_dwt(cond_1)
        cond_2_dwt = apply_dwt(cond_2)
        cond_3_dwt = apply_dwt(cond_3)
        cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)

        # Source: mean of conditions in wavelet space
        source_dwt = (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0

        # Create class labels for modality conditioning
        batch_size = target.shape[0]
        y = th.full((batch_size,), class_idx, device=device, dtype=th.long)

        # Run flow matching sampling
        output_dwt = self.p_sample_loop(
            model=model,
            shape=source_dwt.shape,
            cond=cond_dwt,
            source=source_dwt,
            y=y,
            num_steps=num_steps,
            device=device,
            progress=progress,
            clip_denoised=clip_denoised,
        )

        # Convert to image domain
        output_img = apply_idwt(output_dwt)

        # Clamp and mask
        output_img = output_img.clamp(0., 1.)
        output_img[cond_1 == 0] = 0  # Zero out non-brain regions

        return output_img, target
