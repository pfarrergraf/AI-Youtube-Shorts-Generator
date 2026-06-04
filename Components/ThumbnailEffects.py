from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops, ImageFilter

from Components.ThumbnailV2 import _detect_face_box, _extract_subject_rgba


@dataclass(frozen=True)
class OutlinePreset:
    stroke_color: tuple[int, int, int, int]
    stroke_width: int
    glow_color: tuple[int, int, int, int]
    glow_radius: int
    shadow_color: tuple[int, int, int, int]
    shadow_blur: int
    shadow_offset: tuple[int, int]


OUTLINE_PRESETS: dict[str, OutlinePreset] = {
    "creator_white": OutlinePreset(
        stroke_color=(255, 255, 255, 255),
        stroke_width=14,
        glow_color=(255, 236, 196, 182),
        glow_radius=16,
        shadow_color=(0, 0, 0, 164),
        shadow_blur=18,
        shadow_offset=(10, 18),
    ),
    "creator_blue": OutlinePreset(
        stroke_color=(120, 216, 255, 255),
        stroke_width=12,
        glow_color=(90, 188, 255, 146),
        glow_radius=14,
        shadow_color=(4, 10, 18, 156),
        shadow_blur=16,
        shadow_offset=(8, 16),
    ),
    "sermon_gold": OutlinePreset(
        stroke_color=(255, 214, 138, 255),
        stroke_width=13,
        glow_color=(255, 203, 96, 168),
        glow_radius=18,
        shadow_color=(18, 8, 4, 160),
        shadow_blur=20,
        shadow_offset=(9, 18),
    ),
}


class BackgroundRemovalProvider:
    name = "base"

    def extract_subject(self, frame_bgr):
        raise NotImplementedError


class GrabCutBackgroundRemovalProvider(BackgroundRemovalProvider):
    name = "grabcut_local"

    def extract_subject(self, frame_bgr):
        face_box = _detect_face_box(frame_bgr)
        speaker_rgba, coverage = _extract_subject_rgba(frame_bgr, face_box)
        return speaker_rgba, face_box, coverage


def get_background_removal_provider(name: str | None = None) -> BackgroundRemovalProvider:
    provider_name = str(name or "grabcut_local").strip().lower()
    if provider_name == "grabcut_local":
        return GrabCutBackgroundRemovalProvider()
    raise ValueError(f"Unsupported background removal provider: {provider_name}")


def crop_to_alpha(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return image.crop(bbox) if bbox else image


def add_speaker_outline(subject_rgba: Image.Image, preset_name: str = "creator_white") -> Image.Image:
    preset = OUTLINE_PRESETS[preset_name]
    subject = subject_rgba.convert("RGBA")
    alpha = subject.getchannel("A")

    max_filter = max(3, preset.stroke_width * 2 + 1)
    if max_filter % 2 == 0:
        max_filter += 1
    expanded_alpha = alpha.filter(ImageFilter.MaxFilter(max_filter))
    outline_mask = ImageChops.subtract(expanded_alpha, alpha)
    glow_mask = expanded_alpha.filter(ImageFilter.GaussianBlur(radius=preset.glow_radius))
    shadow_mask = expanded_alpha.filter(ImageFilter.GaussianBlur(radius=preset.shadow_blur))

    pad = max(
        preset.stroke_width * 3,
        preset.glow_radius * 3,
        preset.shadow_blur * 3,
        abs(preset.shadow_offset[0]) + abs(preset.shadow_offset[1]) + 16,
    )
    canvas = Image.new("RGBA", (subject.width + pad * 2, subject.height + pad * 2), (0, 0, 0, 0))
    subject_pos = (pad, pad)

    shadow_layer = Image.new("RGBA", subject.size, preset.shadow_color)
    shadow_layer.putalpha(shadow_mask.point(lambda value: min(255, int(value * preset.shadow_color[3] / 255))))
    canvas.alpha_composite(
        shadow_layer,
        (subject_pos[0] + preset.shadow_offset[0], subject_pos[1] + preset.shadow_offset[1]),
    )

    glow_layer = Image.new("RGBA", subject.size, preset.glow_color)
    glow_layer.putalpha(glow_mask.point(lambda value: min(255, int(value * preset.glow_color[3] / 255))))
    canvas.alpha_composite(glow_layer, subject_pos)

    outline_layer = Image.new("RGBA", subject.size, preset.stroke_color)
    outline_layer.putalpha(outline_mask.point(lambda value: min(255, int(value * preset.stroke_color[3] / 255))))
    canvas.alpha_composite(outline_layer, subject_pos)
    canvas.alpha_composite(subject, subject_pos)
    return crop_to_alpha(canvas)
