"""GPU (torch) equivalents of the reference filters in src/filters_cpu.py (assignment A4).

All functions take a (N,3,H,W) float tensor on the GPU and process the whole
batch at once. Tolerances are defined in tests/test_cpu_gpu_equivalence.py.
"""
import torch
import torch.nn.functional as F


def gaussian_blur_gpu(t, ksize=5, sigma=1.0):
    """TODO(A4.7): Gaussian blur of a batch.

    Hint: build the 1D kernel, take the outer product for a 2D kernel, and apply it
    to all channels with F.conv2d(..., groups=3) after reflect padding
    (OpenCV's default border is reflect-101, which equals torch "reflect").
    Even ksize must be bumped to the next odd value like filters_cpu.gaussian_blur.
    Stretch goal: use two separable 1D passes and measure whether they are faster.
    """
    raise NotImplementedError("TODO(A4.7): gaussian_blur_gpu")


def median_filter_gpu(t, ksize=5):
    """TODO(A4.8): median filter of a batch.

    Hint: pad with mode="replicate" (what cv2.medianBlur does), extract all
    ksize*ksize neighbourhoods with F.unfold, take the median along that axis.
    F.unfold on (N,3,H,W) returns (N, 3*k*k, L): reshape it to (N,3,k*k,H,W) first.
    Note the memory cost: ksize^2 times the input size. Report it.
    """
    raise NotImplementedError("TODO(A4.8): median_filter_gpu")


# TODO(A4.11): pick ONE of the filters above and implement a second GPU version with a
# different tool (CuPy cupyx.scipy.ndimage, a Numba CUDA kernel, or separable convolutions).
# Put it in this file under a clear name and compare both versions against the CPU
# reference in speed and in maximum absolute difference. Which would you ship, and why?


def green_channel_gpu(t):
    """TODO(A4.9): the green channel as a (N,1,H,W) tensor (a view, no copy)."""
    raise NotImplementedError("TODO(A4.9): green_channel_gpu")


def contrast_stretch_gpu(t):
    """TODO(A4.10): per-image contrast stretch using the 1st and 99th percentiles
    of all pixels of that image (same definition as filters_cpu.contrast_stretch).

    Hint: flatten each image and use torch.quantile along dim=1.
    """
    raise NotImplementedError("TODO(A4.10): contrast_stretch_gpu")
