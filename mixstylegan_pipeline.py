"""
MixStyleGAN pipeline — model loading, preprocessing, and generation.

The pipeline:
    Stable Diffusion 1.5
    + ControlNet-Canny (structure lock)
    + ControlNet-Tile (color and composition preservation)
    + 2x IP-Adapter base (spatial style mixing)

Designed to be imported by app.py (Gradio UI) or used directly in a notebook.
The model graph is loaded lazily on first call to load_pipeline() so this
module imports cleanly on machines without CUDA.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

# These imports require a CUDA-capable environment to actually run, but the
# imports themselves are fine on any platform.
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.image_processor import IPAdapterMaskProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_RES = 512        # SD 1.5 native resolution; shorter side targets this
MAX_SIDE = 768        # cap longer side to avoid duplicate-subject artifacts
SIZE_MULTIPLE = 8     # SD requires height/width to be multiples of 8

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_CANNY = "lllyasviel/sd-controlnet-canny"
CONTROLNET_TILE = "lllyasviel/control_v11f1e_sd15_tile"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.safetensors"

# ---------------------------------------------------------------------------
# Module-level pipeline (lazy)
# ---------------------------------------------------------------------------

_pipe: StableDiffusionControlNetPipeline | None = None
_mask_processor: IPAdapterMaskProcessor | None = None
_device: str = "cuda"
_dtype: torch.dtype = torch.float16


def load_pipeline(
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> StableDiffusionControlNetPipeline:
    """Load all models. First call downloads ~3.5 GB and caches them.

    If `device` is None, auto-detects: "cuda" if available, else "cpu".
    If `dtype` is None, picks float16 for CUDA, float32 for CPU.
    CPU inference works but takes 3-5 minutes per image.
    """
    global _pipe, _mask_processor, _device, _dtype
    if _pipe is not None:
        return _pipe

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' explicitly requested but CUDA isn't available. "
            "Pass device='cpu' to fall back, or run on a CUDA-capable host."
        )

    print(f"[MixStyleGAN] Loading pipeline on {device} with dtype {dtype}")
    _device = device
    _dtype = dtype

    controlnet_canny = ControlNetModel.from_pretrained(
        CONTROLNET_CANNY, torch_dtype=dtype,
    )
    controlnet_tile = ControlNetModel.from_pretrained(
        CONTROLNET_TILE, torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL,
        controlnet=[controlnet_canny, controlnet_tile],
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=["models", "models"],
        weight_name=[IP_ADAPTER_WEIGHT, IP_ADAPTER_WEIGHT],
    )
    pipe.set_progress_bar_config(disable=True)

    _pipe = pipe
    _mask_processor = IPAdapterMaskProcessor()
    return _pipe


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def target_size(
    image: Image.Image,
    base: int = BASE_RES,
    max_side: int = MAX_SIDE,
    multiple: int = SIZE_MULTIPLE,
) -> tuple[int, int]:
    """Compute (W, H) for generation — aspect-preserving, multiples of 8,
    short side near `base`, long side capped at `max_side`."""
    w, h = image.size
    short = min(w, h)
    scale = base / short
    new_w = int(round(w * scale / multiple) * multiple)
    new_h = int(round(h * scale / multiple) * multiple)
    longer = max(new_w, new_h)
    if longer > max_side:
        cap = max_side / longer
        new_w = max(multiple, int(round(new_w * cap / multiple) * multiple))
        new_h = max(multiple, int(round(new_h * cap / multiple) * multiple))
    return new_w, new_h


def make_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Canny edge map as a 3-channel PIL image (ControlNet expects RGB)."""
    arr = np.array(image)
    edges = cv2.Canny(arr, int(low), int(high))
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def extract_mask_from_editor(
    editor_value: dict | None,
    target_w: int,
    target_h: int,
) -> tuple[Image.Image | None, Image.Image | None]:
    """Pull a binary mask from a gr.ImageEditor `value` dict.

    Returns (mask_a, mask_b) where mask_a is the user-painted region (white)
    and mask_b is its inverse. Returns (None, None) if the user didn't paint
    anything — caller should fall back to global-mix mode.
    """
    if editor_value is None:
        return None, None
    layers = editor_value.get("layers", [])
    if not layers:
        return None, None
    layer = layers[0]
    if layer is None:
        return None, None
    layer = layer.convert("RGBA").resize((target_w, target_h))
    alpha = np.array(layer)[..., 3]
    if alpha.max() == 0:
        return None, None
    mask_a = (alpha > 0).astype(np.uint8) * 255
    mask_b = 255 - mask_a
    return Image.fromarray(mask_a), Image.fromarray(mask_b)


def full_mask(target_w: int, target_h: int) -> Image.Image:
    """All-white mask matching target size — used in global-mix mode where
    both IP-Adapters apply across the entire image."""
    return Image.fromarray(np.full((target_h, target_w), 255, dtype=np.uint8))


def preview_canny(content_image: Image.Image | None, low: int, high: int) -> Image.Image | None:
    """Live-preview hook for the Gradio UI: regenerate the Canny edge map
    whenever the threshold sliders change."""
    if content_image is None:
        return None
    w, h = target_size(content_image)
    content = content_image.convert("RGB").resize((w, h))
    return make_canny(content, int(low), int(high))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    content_image: Image.Image,
    mask_editor: dict | None,
    style_a: Image.Image,
    style_b: Image.Image,
    weight_a: float,
    weight_b: float,
    prompt: str,
    steps: int,
    controlnet_scale: float,
    guidance: float,
    canny_low: int,
    canny_high: int,
    tile_scale: float,
    seed: int,
):
    """Run the full pipeline. Returns (result, canny_debug, mask_debug, info)."""
    pipe = load_pipeline(_device, _dtype)
    if content_image is None or style_a is None or style_b is None:
        # Caller (Gradio) raises a friendlier error; here we just bail.
        raise ValueError("content_image, style_a, and style_b are all required")

    original_size = content_image.size  # (W, H) at full input resolution
    w, h = target_size(content_image)

    content = content_image.convert("RGB").resize((w, h))
    canny = make_canny(content, canny_low, canny_high)

    mask_a, mask_b = extract_mask_from_editor(mask_editor, w, h)
    spatial_mode = mask_a is not None
    if not spatial_mode:
        mask_a = mask_b = full_mask(w, h)

    masks_tensor = _mask_processor.preprocess([mask_a, mask_b], height=h, width=w)
    ip_adapter_masks = [masks_tensor[i:i + 1] for i in range(2)]

    pipe.set_ip_adapter_scale([float(weight_a), float(weight_b)])
    generator = torch.Generator(device=_device).manual_seed(int(seed))

    result = pipe(
        prompt=prompt or "a painting",
        image=[canny, content],
        ip_adapter_image=[style_a.convert("RGB"), style_b.convert("RGB")],
        cross_attention_kwargs={"ip_adapter_masks": ip_adapter_masks},
        num_inference_steps=int(steps),
        controlnet_conditioning_scale=[float(controlnet_scale), float(tile_scale)],
        guidance_scale=float(guidance),
        height=h,
        width=w,
        generator=generator,
    ).images[0]

    # Lanczos resize to the input's exact dimensions so output matches input
    # pixel-for-pixel. Note: this is interpolation, not super-resolution.
    result = result.resize(original_size, Image.LANCZOS)

    label = "spatial" if spatial_mode else "global"
    info = (
        f"**Mode:** {label}  •  "
        f"**Generated at:** {w}x{h}  •  "
        f"**Output at:** {original_size[0]}x{original_size[1]}"
    )
    return result, canny, mask_a, info
