"""
A script for sampling from a diffusion model for paired image-to-image translation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F

sys.path.append(".")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'

    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='auto')

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12,
                                     shuffle=False,)

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for batch in iter(datal):
        missing = batch['missing'][0]
        print("Missing modality: {}".format(missing))

        if missing == 't1n':
            selected_model_path = '' # INSERT PATH TO 't1n' GENERATING MODEL (weights available on HuggingFace)
        elif missing == 't1c':
            selected_model_path = '' # INSERT PATH TO 't1c' GENERATING MODEL (weights available on HuggingFace)
        elif missing == 't2w':
            selected_model_path = '' # INSERT PATH TO 't2w' GENERATING MODEL (weights available on HuggingFace)
        elif missing == 't2f':
            selected_model_path = '' # INSERT PATH TO 't2f' GENERATING MODEL (weights available on HuggingFace)

        logger.log("Load model from: {}".format(selected_model_path))
        model.load_state_dict(dist_util.load_state_dict(selected_model_path, map_location="cpu"))
        model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices

        batch['t1n'] = batch['t1n'].to(dist_util.dev())
        batch['t1c'] = batch['t1c'].to(dist_util.dev())
        batch['t2w'] = batch['t2w'].to(dist_util.dev())
        batch['t2f'] = batch['t2f'].to(dist_util.dev())

        subj = batch['subj'][0].split('pseudo_validation_v2/')[1][:19]
        print(subj)

        miss_name = args.data_dir + '/' + subj + '/' + subj +'-' + missing
        print(miss_name)

        if missing == 't1n':
            cond_1 = batch['t1c']  # condition
            cond_2 = batch['t2w']  # condition
            cond_3 = batch['t2f']  # condition
            header = nib.load(batch['filedict']['t1c'][0]).header

        elif missing == 't1c':
            cond_1 = batch['t1n']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
            header = nib.load(batch['filedict']['t1n'][0]).header

        elif missing == 't2w':
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2f']
            header = nib.load(batch['filedict']['t1n'][0]).header

        elif missing == 't2f':
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2w']
            header = nib.load(batch['filedict']['t1n'][0]).header

        else:
            print("This contrast can't be synthesized.")

        # Conditioning vector
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
        cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_2)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_3)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)



        # Noise
        noise = th.randn(args.batch_size, 8, 112, 112, 80).to(dist_util.dev())

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs)

        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample[sample <= 0.04] = 0

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        # Pad/Crop to original resolution
        pad_sample = F.pad(sample, (0, 0, 8, 8, 8, 8), mode='constant', value=0)
        sample = pad_sample[:, :, :, :155]


        if len(cond_1.shape) == 5:
            cond_1 = cond_1.squeeze(dim=1)
        if len(cond_2.shape) == 5:
            cond_2 = cond_2.squeeze(dim=1)
        if len(cond_3.shape) == 5:
            cond_3 = cond_3.squeeze(dim=1)


        for i in range(sample.shape[0]):
            output_name = miss_name +'.nii.gz'
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], None, header)
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')




def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        contr="",
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

















