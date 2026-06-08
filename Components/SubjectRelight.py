"""
Optional generative relighting for speaker cutouts via IC-Light (SD 1.5 FC).

Heavy dependencies (torch, diffusers, ~6 GB weights on first run) load lazily —
the base thumbnail pipeline MUST keep working without this module. Callers wrap
relight_subject() in try/except and fall back to the procedural grade.

Note: this AI-alters the speaker's appearance. Only enabled via the explicit
--relight flag, never by default.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ICLIGHT_REPO = "lllyasviel/ic-light"
ICLIGHT_FILE = "iclight_sd15_fc.safetensors"

DEFAULT_PROMPT = (
    "man on stage, cinematic stage lighting, soft warm key light from above, "
    "colored rim light, dramatic atmosphere, professional photography, detailed face, 8k"
)
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, cropped, worst quality, deformed face, plastic skin"

_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    import safetensors.torch
    from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
    from huggingface_hub import hf_hub_download

    # img2img: the original pixels anchor the speaker's identity — pure t2i
    # conditioning regenerates the person (different face), which is unacceptable.
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD15_REPO,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    unet = pipe.unet
    # IC-Light FC: conv_in takes 8 channels (4 noise + 4 foreground latent)
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            8,
            unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        ).to(device=unet.conv_in.weight.device, dtype=unet.conv_in.weight.dtype)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    unet.config["in_channels"] = 8

    original_forward = unet.forward

    def hooked_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        kwargs["cross_attention_kwargs"] = {}
        return original_forward(
            torch.cat([sample, c_concat], dim=1), timestep, encoder_hidden_states, **kwargs
        )

    unet.forward = hooked_forward

    # IC-Light ships offset weights: merged = base + offset
    offset_path = hf_hub_download(ICLIGHT_REPO, ICLIGHT_FILE)
    offset_sd = safetensors.torch.load_file(offset_path)
    unet_sd = unet.state_dict()
    merged = {
        key: unet_sd[key] + offset_sd[key].to(unet_sd[key]) if key in offset_sd else unet_sd[key]
        for key in unet_sd
    }
    unet.load_state_dict(merged)

    pipe = pipe.to("cuda")
    _PIPELINE = pipe
    return pipe


def _fit_dims(width: int, height: int, long_side: int = 768) -> tuple[int, int]:
    scale = long_side / float(max(width, height))
    return (max(64, int(width * scale) // 64 * 64), max(64, int(height * scale) // 64 * 64))


def relight_subject(
    subject_rgba: Image.Image,
    prompt: str | None = None,
    *,
    steps: int = 30,
    guidance_scale: float = 2.0,
    strength: float = 0.30,
    seed: int = 12345,
) -> Image.Image:
    """Relight the cutout with IC-Light (img2img, identity-preserving).
    Returns RGBA in the original size with the original alpha re-applied."""
    import torch

    pipe = _load_pipeline()
    subject = subject_rgba.convert("RGBA")
    orig_size = subject.size
    alpha = subject.getchannel("A")

    work_w, work_h = _fit_dims(*orig_size)
    gray_bg = Image.new("RGBA", orig_size, (127, 127, 127, 255))
    fg = Image.alpha_composite(gray_bg, subject).convert("RGB").resize(
        (work_w, work_h), Image.Resampling.LANCZOS
    )

    fg_np = (np.asarray(fg, dtype=np.float32) / 127.5) - 1.0
    fg_tensor = torch.from_numpy(fg_np.transpose(2, 0, 1)).unsqueeze(0)
    fg_tensor = fg_tensor.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        concat_conds = pipe.vae.encode(fg_tensor).latent_dist.mode() * pipe.vae.config.scaling_factor

    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt or DEFAULT_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=fg,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        cross_attention_kwargs={"concat_conds": concat_conds},
    ).images[0]

    relit = result.resize(orig_size, Image.Resampling.LANCZOS).convert("RGBA")
    relit.putalpha(alpha)
    print("[SubjectRelight] IC-Light relight applied.")
    return relit
