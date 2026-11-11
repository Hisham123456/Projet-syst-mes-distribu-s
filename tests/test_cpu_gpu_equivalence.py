import os
import numpy as np
import pytest
import torch
import yaml

torch.manual_seed(0)

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def sample_batch(n=4, h=224, w=224):
    rng = np.random.default_rng(1)
    imgs = rng.uniform(0.2, 0.9, (n, h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (min(h, w) // 2 - 2) ** 2
    imgs *= mask[..., None]
    return imgs


@cuda
def test_circular_crop_equivalence():
    from src import preprocess_cpu, preprocess_gpu
    imgs = sample_batch(2)
    cpu = np.stack([preprocess_cpu.circular_crop(i) for i in imgs])
    gpu = preprocess_gpu.circular_crop_gpu(
        torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()).permute(0, 2, 3, 1).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-5)


@cuda
def test_normalize_equivalence():
    from src import preprocess_cpu, preprocess_gpu
    imgs = sample_batch(2)
    cpu = np.stack([preprocess_cpu.normalize(i) for i in imgs])
    gpu = preprocess_gpu.normalize_gpu(
        torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()).permute(0, 2, 3, 1).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-5)


@cuda
def test_clahe_equivalence_within_tolerance():
    """GPU CLAHE is a reimplementation; require structural equivalence.

    Tolerance is deliberately loose (correlation on luminance) because the
    CPU (OpenCV) and GPU (torch) histogram pipelines differ in binning detail.
    """
    from src import preprocess_cpu, preprocess_gpu
    imgs = sample_batch(2)
    for i in range(2):
        cpu = preprocess_cpu.apply_clahe(imgs[i]).mean(-1)
        gpu = preprocess_gpu.clahe_gpu(
            torch.from_numpy(imgs[i]).permute(2, 0, 1)[None].cuda()
        )[0].permute(1, 2, 0).cpu().numpy().mean(-1)
        assert np.corrcoef(cpu.ravel(), gpu.ravel())[0, 1] > 0.95


@cuda
def test_gaussian_filter_equivalence():
    from src import filters_cpu, filters_gpu
    img = sample_batch(1)[0]
    cpu = filters_cpu.gaussian_blur(img, ksize=5, sigma=1.0)
    gpu = filters_gpu.gaussian_blur_gpu(
        torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
    )[0].permute(1, 2, 0).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=2e-2)


@cuda
def test_median_filter_equivalence():
    from src import filters_cpu, filters_gpu
    img = sample_batch(1)[0]
    cpu = filters_cpu.median_filter(img, ksize=5)
    gpu = filters_gpu.median_filter_gpu(
        torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
    )[0].permute(1, 2, 0).cpu().numpy()
    frac_diff = (np.abs(cpu - gpu) > (1.0 / 255)).mean()
    assert frac_diff < 0.02               # <=2% pixels may differ by one gray level


@cuda
def test_clahe_batched_matches_single_image():
    """Batched CLAHE must give each image the same result as processing it alone."""
    from src import preprocess_gpu
    t = torch.from_numpy(sample_batch(3)).permute(0, 3, 1, 2).cuda()
    batched = preprocess_gpu.clahe_gpu(t)
    for i in range(3):
        single = preprocess_gpu.clahe_gpu(t[i:i + 1])
        assert torch.allclose(batched[i:i + 1], single, atol=1e-5)


@cuda
def test_contrast_stretch_equivalence():
    from src import filters_cpu, filters_gpu
    img = sample_batch(1)[0]
    cpu = filters_cpu.contrast_stretch(img)
    gpu = filters_gpu.contrast_stretch_gpu(
        torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
    )[0].permute(1, 2, 0).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-3)


@cuda
def test_uint8_batch_pipeline_matches_cpu_without_clahe():
    """Full GPU batch path from uint8 (as the loader delivers it) equals the CPU
    path exactly when only the deterministic steps (crop, normalize) are on."""
    from src import preprocess_cpu, preprocess_gpu
    cfg = {"preprocess": {"clahe": False, "circular_crop": True, "normalize": True, "gamma": 1.0}}
    u8 = (sample_batch(2) * 255).astype(np.uint8)
    cpu = np.stack([preprocess_cpu.preprocess_image(i.astype(np.float32) / 255.0, cfg) for i in u8])
    gpu = preprocess_gpu.preprocess_batch_gpu(
        torch.from_numpy(u8).permute(0, 3, 1, 2).contiguous(), cfg).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-5)


@cuda
def test_augment_gpu_shape_range_and_per_image_randomness():
    from src import preprocess_gpu
    torch.manual_seed(0)
    t = torch.from_numpy(sample_batch(8)).permute(0, 3, 1, 2).cuda()
    out = preprocess_gpu.augment_gpu(t)
    assert out.shape == t.shape
    assert out.min() >= 0 and out.max() <= 1 + 1e-5
    assert not torch.allclose(out, t)
    # different images must not all receive the same transform
    assert not torch.allclose(out[0] - t[0], out[1] - t[1])
