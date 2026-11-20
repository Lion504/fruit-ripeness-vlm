"""Procedural backgrounds, shrink-and-paste, white balance, blur+JPEG. Shared by the hard-set builder
and the training augmentation. Peel colour is never touched except by the test-only white-balance shift."""

import io

import numpy as np
from PIL import Image, ImageFilter

NAMED = {
    "brown": (121, 85, 56),
    "beige": (222, 205, 170),
    "yellow": (232, 205, 60),
    "white": (245, 245, 245),
    "darkgrey": (55, 55, 58),
}


def solid(rng, size, color=None):
    if color is None:
        color = rng.integers(0, 256, 3)
    return np.tile(np.asarray(color, np.float32), (size, size, 1))


def stripes(rng, size, color=None):
    """Wood-like stripes: sinusoid along a random axis with warped phase and grain noise."""
    base = np.asarray(
        color if color is not None else NAMED["brown"], np.float32
    ) * rng.uniform(0.7, 1.2)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    u = xx * np.cos(ang) + yy * np.sin(ang)
    f = rng.uniform(0.05, 0.2)
    phase = np.sin(yy * 0.03 + rng.uniform(0, 6)) * 6
    s = 0.5 + 0.5 * np.sin((u + phase) * f * 2 * np.pi / 6.0)
    img = base[None, None] * (0.75 + 0.35 * s[..., None]) + rng.normal(
        0, 6, (size, size, 1)
    )
    return img


def noise(rng, size, color=None):
    base = np.asarray(
        color if color is not None else rng.integers(30, 230, 3), np.float32
    )
    n = rng.normal(0, 28, (size // 4, size // 4, 1)).repeat(3, 2) * rng.uniform(
        0.5, 1.2, 3
    )
    n = (
        np.asarray(
            Image.fromarray(np.clip(n + 128, 0, 255).astype(np.uint8)).resize(
                (size, size), Image.BILINEAR
            ),
            np.float32,
        )
        - 128
    )
    return base[None, None] + n + rng.normal(0, 8, (size, size, 3))


def make_bg(rng, size, kind, color=None):
    f = {"solid": solid, "stripes": stripes, "noise": noise}[kind]
    return Image.fromarray(np.clip(f(rng, size, color), 0, 255).astype(np.uint8))


def paste_on_bg(img, rng, kind, color=None, scale=None, center=True):
    s = scale if scale is not None else rng.uniform(0.5, 0.8)
    n = img.width
    m = max(8, int(n * s))
    bg = make_bg(rng, n, kind, color)
    if center:
        ox = oy = (n - m) // 2
    else:
        ox, oy = int(rng.integers(0, n - m + 1)), int(rng.integers(0, n - m + 1))
    bg.paste(img.resize((m, m), Image.BILINEAR), (ox, oy))
    return bg


def white_balance(img, warm):
    a = np.asarray(img, np.float32)
    g = np.array([1.14, 1.0, 0.86] if warm else [0.88, 1.0, 1.14], np.float32)
    return Image.fromarray(np.clip(a * g, 0, 255).astype(np.uint8))


def blur_jpeg(img, q=50, radius=1.2):
    b = io.BytesIO()
    img.filter(ImageFilter.GaussianBlur(radius)).save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


class RandomBgPaste:
    """Training augmentation: with prob p shrink the photo (scale 0.5-1) and paste at a random offset onto a
    random solid/stripe/noise background. The whole banana stays inside the frame."""

    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, img):
        import torch

        seed = int(torch.randint(0, 2**31 - 1, (1,)))
        rng = np.random.default_rng(seed)
        if rng.random() > self.p:
            return img
        kind = rng.choice(["solid", "stripes", "noise"], p=[0.5, 0.25, 0.25])
        color = None
        if kind != "noise" and rng.random() < 0.5:
            c = np.asarray(
                list(NAMED.values())[rng.integers(0, 5)], np.float32
            ) * rng.uniform(0.8, 1.15) + rng.normal(0, 10, 3)
            color = np.clip(c, 0, 255)
        return paste_on_bg(
            img, rng, kind, color, scale=rng.uniform(0.5, 1.0), center=False
        )


def _kelvin_rgb(k):
    """Approximate blackbody colour of a light source at k Kelvin (curve fit, 1000-40000 K)."""
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = (
        99.4708025861 * np.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    b = (
        255
        if t >= 66
        else (0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    )
    return np.clip(np.array([r, g, b], np.float64), 1, 255)


def kelvin_gains(k, strength=0.5):
    """Per-channel gains that make a scene look lit by a k Kelvin source instead of 6500 K (green gain 1)."""
    g = (_kelvin_rgb(k) / _kelvin_rgb(6500)) ** strength
    return g / g[1]


class WBJitter:
    """Training augmentation: with prob p apply Kelvin-style gains (k log-uniform 3000-7500) to the whole image."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        import torch

        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        if rng.random() > self.p:
            return img
        k = float(np.exp(rng.uniform(np.log(3000), np.log(7500))))
        return Image.fromarray(
            np.clip(
                np.asarray(img, np.float32) * kelvin_gains(k)[None, None], 0, 255
            ).astype(np.uint8)
        )
