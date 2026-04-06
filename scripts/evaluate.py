"""
Evaluate synthesized images: PSNR, SSIM, MSE.
"""

import argparse
import numpy as np
import os
import nibabel as nib
from glob import glob
from tqdm import tqdm


def load_nifti(path):
    """Load and normalize NIfTI image."""
    img = nib.load(path).get_fdata()
    # Handle potential NaN values
    img = np.nan_to_num(img, nan=0.0)
    # Clip to positive values
    img = np.clip(img, 0, None)
    # Normalize to [0, 1]
    if img.max() > 0:
        img = img / img.max()
    return img.astype(np.float32)


def compute_psnr(pred, gt, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: predicted image
        gt: ground truth image
        max_val: maximum possible value

    Returns:
        PSNR in dB
    """
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(pred, gt, data_range=1.0):
    """
    Compute Structural Similarity Index (SSIM).

    Uses 3D implementation for volumetric data.

    Args:
        pred: predicted image
        gt: ground truth image
        data_range: data range of input images

    Returns:
        SSIM value
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        # For 3D data, compute SSIM slice-wise and average
        # This matches common practice in medical imaging
        return ssim(gt, pred, data_range=data_range)
    except ImportError:
        # Fallback implementation
        return _ssim_fallback(pred, gt, data_range)


def _ssim_fallback(pred, gt, data_range=1.0):
    """
    Fallback SSIM implementation when skimage is not available.
    Computes SSIM using the original Wang et al. formula.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)
    sigma_pred = np.std(pred)
    sigma_gt = np.std(gt)
    sigma_pred_gt = np.mean((pred - mu_pred) * (gt - mu_gt))

    numerator = (2 * mu_pred * mu_gt + C1) * (2 * sigma_pred_gt + C2)
    denominator = (mu_pred ** 2 + mu_gt ** 2 + C1) * (sigma_pred ** 2 + sigma_gt ** 2 + C2)

    return numerator / denominator


def compute_mse(pred, gt):
    """Compute Mean Squared Error."""
    return np.mean((pred - gt) ** 2)


def compute_mae(pred, gt):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(pred - gt))


def compute_metrics_masked(pred, gt, mask=None):
    """
    Compute metrics only on masked (brain) region.

    Args:
        pred: predicted image
        gt: ground truth image
        mask: binary mask (if None, use non-zero region of gt)

    Returns:
        dict of metrics
    """
    if mask is None:
        # Use non-zero region as mask
        mask = gt > 0

    pred_masked = pred[mask]
    gt_masked = gt[mask]

    if len(pred_masked) == 0:
        return {'psnr': 0, 'ssim': 0, 'mse': 0, 'mae': 0}

    # Compute MSE and MAE on masked region
    mse = np.mean((pred_masked - gt_masked) ** 2)
    mae = np.mean(np.abs(pred_masked - gt_masked))

    # PSNR from MSE
    max_val = max(gt_masked.max(), 1.0)
    if mse > 0:
        psnr = 10 * np.log10(max_val ** 2 / mse)
    else:
        psnr = float('inf')

    # SSIM on full volume (not masked)
    ssim = compute_ssim(pred, gt, data_range=max_val)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
        'mae': mae
    }


def find_matching_files(pred_dir, gt_dir, contr, pred_pattern=None, gt_pattern=None):
    """
    Find matching prediction and ground truth files.

    Args:
        pred_dir: directory containing predictions
        gt_dir: directory containing ground truth
        contr: contrast being evaluated
        pred_pattern: glob pattern for predictions
        gt_pattern: glob pattern for ground truth

    Returns:
        list of (pred_path, gt_path) tuples
    """
    pairs = []

    # Default patterns
    if pred_pattern is None:
        pred_pattern = f"**/sample_{contr}.nii.gz"
    if gt_pattern is None:
        gt_pattern = f"**/target_{contr}.nii.gz"

    # Find all prediction files
    pred_files = sorted(glob(os.path.join(pred_dir, pred_pattern), recursive=True))

    for pred_file in pred_files:
        # Extract subject ID from path
        rel_path = os.path.relpath(pred_file, pred_dir)
        subj_dir = os.path.dirname(rel_path)

        # Look for matching ground truth
        gt_file = os.path.join(pred_dir, subj_dir, f"target_{contr}.nii.gz")

        if os.path.exists(gt_file):
            pairs.append((pred_file, gt_file))
        else:
            # Try alternative GT location
            gt_file_alt = os.path.join(gt_dir, subj_dir, f"*{contr}*.nii.gz")
            gt_matches = glob(gt_file_alt)
            if gt_matches:
                pairs.append((pred_file, gt_matches[0]))

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Evaluate synthesized MRI images')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory with predictions')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Directory with ground truth (optional, uses target files from pred_dir if not specified)')
    parser.add_argument('--contr', type=str, default='t2f',
                        help='Contrast to evaluate (t1n, t1c, t2w, t2f)')
    parser.add_argument('--output', type=str, default='metrics.txt',
                        help='Output file for metrics')
    parser.add_argument('--use_mask', action='store_true',
                        help='Compute metrics only on brain region')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample metrics')
    args = parser.parse_args()

    # Find file pairs
    if args.gt_dir is None:
        args.gt_dir = args.pred_dir

    pairs = find_matching_files(args.pred_dir, args.gt_dir, args.contr)

    if len(pairs) == 0:
        print(f"Error: No matching files found in {args.pred_dir}")
        print(f"Looking for pattern: **/sample_{args.contr}.nii.gz")
        return

    print(f"Found {len(pairs)} sample pairs")

    # Compute metrics
    results = {
        'psnr': [],
        'ssim': [],
        'mse': [],
        'mae': []
    }

    for pred_file, gt_file in tqdm(pairs, desc="Evaluating"):
        pred = load_nifti(pred_file)
        gt = load_nifti(gt_file)

        # Ensure same shape
        if pred.shape != gt.shape:
            print(f"Warning: Shape mismatch for {pred_file}")
            print(f"  Pred: {pred.shape}, GT: {gt.shape}")
            continue

        if args.use_mask:
            metrics = compute_metrics_masked(pred, gt)
        else:
            metrics = {
                'psnr': compute_psnr(pred, gt),
                'ssim': compute_ssim(pred, gt),
                'mse': compute_mse(pred, gt),
                'mae': compute_mae(pred, gt)
            }

        for k, v in metrics.items():
            results[k].append(v)

        if args.verbose:
            subj = os.path.basename(os.path.dirname(pred_file))
            print(f"{subj}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, "
                  f"MSE={metrics['mse']:.2e}, MAE={metrics['mae']:.4f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Contrast: {args.contr}")
    print(f"Number of samples: {len(results['psnr'])}")
    print("-" * 60)
    print(f"PSNR:  {np.mean(results['psnr']):7.2f} +/- {np.std(results['psnr']):.2f} dB")
    print(f"SSIM:  {np.mean(results['ssim']):7.4f} +/- {np.std(results['ssim']):.4f}")
    print(f"MSE:   {np.mean(results['mse']):7.2e} +/- {np.std(results['mse']):.2e}")
    print(f"MAE:   {np.mean(results['mae']):7.4f} +/- {np.std(results['mae']):.4f}")
    print("=" * 60)

    # Additional statistics
    print("\nPercentile Statistics:")
    print("-" * 60)
    for metric in ['psnr', 'ssim']:
        vals = results[metric]
        print(f"{metric.upper()}:")
        print(f"  Min:    {np.min(vals):.4f}")
        print(f"  25%:    {np.percentile(vals, 25):.4f}")
        print(f"  Median: {np.median(vals):.4f}")
        print(f"  75%:    {np.percentile(vals, 75):.4f}")
        print(f"  Max:    {np.max(vals):.4f}")

    # Save to file
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.pred_dir, output_path)

    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Contrast: {args.contr}\n")
        f.write(f"Number of samples: {len(results['psnr'])}\n")
        f.write(f"Prediction directory: {args.pred_dir}\n")
        f.write("-" * 60 + "\n")
        f.write(f"PSNR:  {np.mean(results['psnr']):.2f} +/- {np.std(results['psnr']):.2f} dB\n")
        f.write(f"SSIM:  {np.mean(results['ssim']):.4f} +/- {np.std(results['ssim']):.4f}\n")
        f.write(f"MSE:   {np.mean(results['mse']):.2e} +/- {np.std(results['mse']):.2e}\n")
        f.write(f"MAE:   {np.mean(results['mae']):.4f} +/- {np.std(results['mae']):.4f}\n")
        f.write("=" * 60 + "\n")

        f.write("\nPer-sample results:\n")
        f.write("-" * 60 + "\n")
        for i, (pred_file, _) in enumerate(pairs):
            if i < len(results['psnr']):
                subj = os.path.basename(os.path.dirname(pred_file))
                f.write(f"{subj}: PSNR={results['psnr'][i]:.2f}, "
                        f"SSIM={results['ssim'][i]:.4f}, "
                        f"MSE={results['mse'][i]:.2e}\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
