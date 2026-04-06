# general settings
GPU=0;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='train';             # train, sample, auto (for automatic missing contrast generation)
DATASET='brats';          # brats
MODEL='unet';             # 'unet'
CONTR='t1n'               # contrast to be generate by the network ('t1n', t1c', 't2w', 't2f') - just relevant during training

# settings for sampling/inference
ITERATIONS=1200;          # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0;         # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="";               # tensorboard dir to be set for the evaluation (displayed at start of training)

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'unet' ]]; then
  echo "MODEL: WDM (U-Net)";
  CHANNEL_MULT=1,2,2,4,4;
  ADDITIVE_SKIP=False;      # Set True to save memory
  BATCH_SIZE=1;
  IMAGE_SIZE=224;
  IN_CHANNELS=32;           # Change to work with different number of conditioning images 8 + 8x (with x number of conditioning images)
  NOISE_SCHED='linear';
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

# some information and overwriting batch size for sampling
# (overwrite in case you want to sample with a higher batch size)
# no need to change for reproducing

if [[ $MODE == 'train' ]]; then
  echo "MODE: training";
  if [[ $DATASET == 'brats' ]]; then
    echo "DATASET: BRATS";
    DATA_DIR=./datasets/BRATS2023/training;
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi

elif [[ $MODE == 'sample' ]]; then
  BATCH_SIZE=1;
  echo "MODE: sampling (image-to-image translation)";
  if [[ $DATASET == 'brats' ]]; then
    echo "DATASET: BRATS";
    DATA_DIR=./datasets/BRATS2023/validation;
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi

elif [[ $MODE == 'auto' ]]; then
  BATCH_SIZE=1;
  echo "MODE: sampling in automatic mode (image-to-image translation)";
  if [[ $DATASET == 'brats' ]]; then
    echo "DATASET: BRATS";
    DATA_DIR=./datasets/BRATS2023/pseudo_validation;
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=${NOISE_SCHED}
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=False
--predict_xstart=True
--contr=${CONTR}
"

TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=100000
--num_workers=12
--num_workers=12
--devices=${GPU}
"
SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--devices=${GPU}
--output_dir=./results/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
"

# run the python scripts
if [[ $MODE == 'train' ]]; then
  python scripts/train.py $TRAIN $COMMON;

elif [[ $MODE == 'sample' ]]; then
  python scripts/sample.py $SAMPLE $COMMON;

elif [[ $MODE == 'auto' ]]; then
  python scripts/sample_auto.py $SAMPLE $COMMON;

else
  echo "MODE NOT FOUND -> Check the supported modes again";
fi
