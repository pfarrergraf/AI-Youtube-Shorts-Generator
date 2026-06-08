from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

from Components import ThumbnailMoveChurch as thumb_mc
from Components.ThumbnailEffects import composite_layer
from Components.ThumbnailMoveChurch import generate_move_church_thumbnail


def _make_subject() -> dict:
    speaker = np.zeros((820, 420, 4), dtype=np.uint8)
    cv2.ellipse(speaker, (210, 230), (170, 205), 0, 0, 360, (182, 136, 104, 255), -1)
    cv2.rectangle(speaker, (112, 410), (308, 800), (42, 48, 60, 255), -1)
    return {
        "speaker_rgba": Image.fromarray(speaker, mode="RGBA"),
        "face_box": (78, 55, 240, 190),
        "coverage": 0.24,
        "provider_used": "test",
        "removal_attempts": [{"provider": "test", "accepted": True}],
        "caption_cleanup": {},
    }


def test_resolve_effect_profile_exposes_high_end_defaults():
    name, config = thumb_mc._resolve_effect_profile("premium")

    assert name == "premium"
    assert config["badge"]["mode"] == "circle"
    assert config["finish"]["grain"] > 0


def test_premium_effect_profile_writes_render_layers_and_badge_report(tmp_path):
    frame = np.zeros((720, 405, 3), dtype=np.uint8)
    frame[:] = (28, 24, 20)
    output = tmp_path / "premium_thumb.png"

    generate_move_church_thumbnail(
        frame,
        title_back="GOTT HAT",
        title_front="EINEN PLAN",
        template="warm_gold",
        fmt="9x16",
        show_decorations=False,
        show_logo=False,
        effect_profile="premium",
        badge_text="MOVE CHURCH",
        badge_mode="circle",
        badge_position="top_right",
        output_path=str(output),
        _precomputed_subject=_make_subject(),
    )

    report_path = output.with_suffix(".thumbnail_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["effect_profile"] == "premium"
    assert any(layer["name"] == "badge" for layer in report["render_layers"])
    assert report["render_layers"][-1]["kind"] == "adjustment_layer"
    assert report["render_layers"][-1]["grain"] > 0.0
    assert output.exists()


def test_path_badge_adds_visible_pixels(monkeypatch):
    font_path = thumb_mc._resolve_mc_font_path("warm_gold") or thumb_mc._find_font()
    assert font_path is not None
    monkeypatch.setattr(thumb_mc, "_resolve_mc_font_path", lambda template=None: font_path)

    canvas = Image.new("RGBA", (1080, 1920), (18, 18, 18, 255))
    before = np.asarray(canvas, dtype=np.uint8)

    after = thumb_mc._add_path_badge(
        canvas,
        template="navy_dark",
        text="MOVE CHURCH",
        mode="arc",
        position="top_right",
    )
    after_arr = np.asarray(after, dtype=np.uint8)

    assert np.any(after_arr != before)


def test_profile_overlay_stack_modifies_canvas_and_records_layers():
    canvas = Image.new("RGBA", (1080, 1920), (16, 18, 24, 255))
    face_box = (420, 210, 220, 260)

    after, layers = thumb_mc._apply_profile_overlay_stack(
        canvas,
        template="warm_gold",
        effect_profile="premium",
        face_box=face_box,
    )

    assert layers
    assert {layer["name"] for layer in layers} >= {"subject_halo", "diagonal_light_sheet", "lower_text_plate"}
    assert np.any(np.asarray(after, dtype=np.uint8) != np.asarray(canvas, dtype=np.uint8))


def test_letter_spacing_tightens_long_words_for_premium_profiles():
    classic = thumb_mc._resolve_letter_spacing("EVANGELIUM", font_size=220, effect_profile="classic")
    premium = thumb_mc._resolve_letter_spacing("EVANGELIUM", font_size=220, effect_profile="premium")
    short = thumb_mc._resolve_letter_spacing("JA", font_size=220, effect_profile="classic")

    assert premium <= classic
    assert short >= classic


def test_spaced_text_measurement_accounts_for_outline_width():
    font = thumb_mc._load_mc_font(120, "warm_gold")

    tight = thumb_mc._measure_spaced_text("GOTT", font, 2, stroke_width=0)
    outlined = thumb_mc._measure_spaced_text("GOTT", font, 2, stroke_width=10)

    assert outlined[0] > tight[0]
    assert outlined[1] >= tight[1]


def test_thumbnail_effects_supports_advanced_blends():
    base = Image.new("RGBA", (8, 8), (120, 120, 120, 255))
    overlay = Image.new("RGBA", (8, 8), (220, 80, 40, 180))

    overlay_result = composite_layer(base, overlay, blend_mode="overlay")
    soft_result = composite_layer(base, overlay, blend_mode="soft_light")
    darken_result = composite_layer(base, overlay, blend_mode="darken")

    base_arr = np.asarray(base, dtype=np.uint8)
    overlay_arr = np.asarray(overlay_result, dtype=np.uint8)
    soft_arr = np.asarray(soft_result, dtype=np.uint8)
    dark_arr = np.asarray(darken_result, dtype=np.uint8)

    assert np.any(overlay_arr != base_arr)
    assert np.any(soft_arr != base_arr)
    assert np.any(dark_arr != base_arr)


def test_custom_layer_stack_supports_shape_and_path_text():
    canvas = Image.new("RGBA", (1080, 1920), (16, 18, 24, 255))
    layer_specs = [
        {
            "name": "glass_plate",
            "kind": "shape",
            "shape": "rounded_rect",
            "box": [80, 1260, 1000, 1810],
            "radius": 48,
            "feather": 26,
            "color": [255, 255, 255, 100],
            "blend_mode": "overlay",
            "opacity": 0.85,
        },
        {
            "name": "arc_badge",
            "kind": "badge",
            "text": "MOVE CHURCH",
            "mode": "arc",
            "center": [840, 220],
            "radius": 138,
            "blend_mode": "alpha",
        },
    ]

    after, meta = thumb_mc.apply_custom_layer_stack(
        canvas,
        layer_specs,
        template="warm_gold",
        default_text="GOTT HAT EINEN PLAN",
    )

    assert {item["name"] for item in meta} == {"glass_plate", "arc_badge"}
    assert np.any(np.asarray(after, dtype=np.uint8) != np.asarray(canvas, dtype=np.uint8))
