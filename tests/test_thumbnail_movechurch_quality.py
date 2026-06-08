import cv2
import numpy as np
import sys

import generate_thumbnail
from Components import ThumbnailMoveChurch as thumb_mc
from Components.ComfyUIBackground import build_background_prompt, _prepare_workflow
from Components.LanguageTasks import _is_invalid_hook
from Components.ThumbnailMoveChurch import (
    _clean_baked_caption_regions,
    _face_overlap_ratio,
    _word_face_overlap_ratio,
    generate_move_church_thumbnail,
)


def test_word_face_overlap_ignores_empty_space_between_text_blocks():
    face_box = (400, 320, 220, 260)
    word_bounds = [
        (50, 60, 520, 160),
        (50, 700, 520, 820),
    ]
    union_box = (50, 60, 520, 820)

    assert _face_overlap_ratio(union_box, face_box) > 0.15
    assert _word_face_overlap_ratio(word_bounds, face_box) == 0.0


def test_word_face_overlap_counts_actual_word_contact():
    face_box = (400, 320, 220, 260)
    word_bounds = [
        (420, 360, 620, 460),
        (50, 700, 520, 820),
    ]

    assert _word_face_overlap_ratio(word_bounds, face_box) > 0.25


def test_clean_baked_caption_regions_inpaints_lower_subtitles():
    frame = np.zeros((720, 405, 3), dtype=np.uint8)
    frame[:] = (20, 16, 14)
    cv2.rectangle(frame, (70, 470), (335, 525), (4, 4, 4), thickness=-1)
    cv2.putText(
        frame,
        "genau",
        (100, 510),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "gesehen",
        (95, 585),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cleaned, info = _clean_baked_caption_regions(frame)

    assert info["applied"] is True
    assert info["mask_coverage"] > 0.01
    assert len(info["boxes"]) >= 1
    assert cleaned[500, 160].mean() < frame[500, 160].mean()


def test_no_subject_fallback_darkens_bright_stage_frame():
    bright_frame = np.full((720, 1280, 3), 242, dtype=np.uint8)

    image = generate_move_church_thumbnail(
        bright_frame,
        title="HELLER HINTERGRUND",
        template="navy_dark",
        fmt="9x16",
        show_decorations=False,
        show_accent_bar=False,
        _precomputed_subject={
            "speaker_rgba": None,
            "face_box": None,
            "coverage": 0.0,
            "provider_used": None,
            "removal_attempts": [{"provider": "test", "accepted": False}],
        },
    ).convert("RGB")

    # Sample a text-free upper-right patch; raw fallback would still be near 242.
    patch = np.asarray(image.crop((720, 120, 1040, 420)), dtype=np.uint8)
    assert float(patch.mean()) < 105.0


def test_bold_minimal_requests_ai_background_without_subject(monkeypatch):
    frame = np.zeros((720, 405, 3), dtype=np.uint8)
    frame[:] = (28, 24, 20)
    called = {}

    def fake_render_ai_background(*, title, template, output_path, size, prompt=None, negative_prompt=None, speaker_name=None, brand_label=None):
        called["title"] = title
        called["template"] = template
        called["output_path"] = output_path
        called["prompt"] = prompt
        called["negative_prompt"] = negative_prompt
        return thumb_mc.Image.new("RGBA", size, (18, 18, 22, 255)), {"backend": "comfyui", "cache_path": None}

    monkeypatch.setattr(thumb_mc, "_render_ai_background", fake_render_ai_background)

    generate_move_church_thumbnail(
        frame,
        title="STANDHAFT IN DER ENDZEIT",
        template="bold_minimal",
        fmt="9x16",
        show_decorations=False,
        show_accent_bar=False,
        output_path=None,
        _precomputed_subject={
            "speaker_rgba": None,
            "face_box": None,
            "coverage": 0.0,
            "provider_used": None,
            "removal_attempts": [{"provider": "test", "accepted": False}],
        },
    )

    assert called["template"] == "bold_minimal"
    assert "endzeit" in called["title"].lower()


def test_upper_face_long_title_keeps_compact_move_church_stack():
    frame = np.zeros((720, 405, 3), dtype=np.uint8)
    frame[:] = (32, 26, 22)

    speaker = np.zeros((900, 450, 4), dtype=np.uint8)
    cv2.ellipse(speaker, (225, 420), (185, 440), 0, 0, 360, (48, 52, 58, 255), -1)
    cv2.ellipse(speaker, (225, 145), (74, 88), 0, 0, 360, (172, 128, 100, 255), -1)

    generate_move_church_thumbnail(
        frame,
        title="GOTT NUTZT WEN ER WILL",
        template="warm_gold",
        fmt="9x16",
        show_decorations=False,
        show_accent_bar=False,
        _precomputed_subject={
            "speaker_rgba": thumb_mc.Image.fromarray(speaker, mode="RGBA"),
            "face_box": (151, 57, 148, 176),
            "coverage": 0.22,
            "provider_used": "test",
            "removal_attempts": [{"provider": "test", "accepted": True}],
        },
    )

    layout = thumb_mc._LAST_LAYOUT_METADATA
    assert layout["composition"] == "upper_face"
    assert layout["vertical_gap_ratio"] <= 0.38
    assert layout["front_body_overlap_ratio"] >= 0.10
    assert layout["front_face_overlap_ratio"] == 0.0
    assert layout["face_overlap_ratio"] == 0.0


def test_head_dominant_cutout_is_scaled_as_closeup():
    frame = np.zeros((720, 405, 3), dtype=np.uint8)
    frame[:] = (30, 24, 20)

    speaker = np.zeros((720, 320, 4), dtype=np.uint8)
    cv2.ellipse(speaker, (160, 230), (150, 210), 0, 0, 360, (166, 118, 92, 255), -1)
    cv2.rectangle(speaker, (96, 430), (224, 710), (40, 44, 52, 255), -1)

    generate_move_church_thumbnail(
        frame,
        title="OFFENBARUNG DER ZUKUNFT",
        template="warm_gold",
        fmt="9x16",
        show_decorations=False,
        show_accent_bar=False,
        _precomputed_subject={
            "speaker_rgba": thumb_mc.Image.fromarray(speaker, mode="RGBA"),
            "face_box": (80, 90, 160, 185),
            "coverage": 0.10,
            "provider_used": "test",
            "removal_attempts": [{"provider": "test", "accepted": True}],
        },
    )

    layout = thumb_mc._LAST_LAYOUT_METADATA
    body_box = layout["body_box"]
    assert layout["closeup_subject"] is True
    assert layout["source_head_span_ratio"] > 0.62
    assert body_box[3] - body_box[1] <= int(1920 * 0.57)


def test_quick_generate_preserves_provider_and_output_api(monkeypatch, tmp_path):
    calls = []

    def fake_generate(source, **kwargs):
        calls.append((source, kwargs))

    monkeypatch.setattr(generate_thumbnail, "generate_move_church_thumbnail", fake_generate)

    output = generate_thumbnail.quick_generate(
        "/tmp/source.mp4",
        title="GOTT IST TREU",
        template="warm_gold",
        output_dir=str(tmp_path),
        provider="rembg",
    )

    assert output == str(tmp_path / "source_warm_gold_9x16.png")
    assert calls[0][1]["bg_removal_provider"] == "rembg"
    assert calls[0][1]["title"] == "GOTT IST TREU"


def test_batch_generate_preserves_back_front_and_provider_api(monkeypatch, tmp_path):
    calls = []

    def fake_generate(source, **kwargs):
        calls.append((source, kwargs))

    monkeypatch.setattr(generate_thumbnail, "generate_move_church_thumbnail", fake_generate)

    results = generate_thumbnail.batch_generate(
        [
            {
                "source": "/tmp/predigt.mp4",
                "title": "EINE WIE KEINE",
                "back": "EINE WIE",
                "front": "KEINE",
                "template": "energy_orange",
            }
        ],
        output_dir=str(tmp_path),
        provider="grabcut_local",
    )

    assert results[0]["output"] == str(tmp_path / "predigt_energy_orange_9x16.png")
    assert calls[0][1]["title_back"] == "EINE WIE"
    assert calls[0][1]["title_front"] == "KEINE"
    assert calls[0][1]["bg_removal_provider"] == "grabcut_local"


def test_cli_single_template_honors_output_dir(monkeypatch, tmp_path):
    calls = []

    def fake_generate(source, **kwargs):
        calls.append((source, kwargs))

    monkeypatch.setattr(generate_thumbnail, "generate_move_church_thumbnail", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_thumbnail.py",
            "--source",
            "/tmp/source.mp4",
            "--title",
            "GOTT IST TREU",
            "--template",
            "warm_gold",
            "--output-dir",
            str(tmp_path),
        ],
    )

    generate_thumbnail._cli()

    assert calls[0][1]["output_path"] == str(tmp_path / "source_warm_gold_9x16.png")
    assert calls[0][1]["title"] == "GOTT IST TREU"


def test_cli_exports_studio_bundle_when_requested(monkeypatch, tmp_path):
    generate_calls = []
    export_calls = []

    def fake_generate(source, **kwargs):
        generate_calls.append((source, kwargs))

    def fake_export(thumbnail_path, **kwargs):
        export_calls.append((thumbnail_path, kwargs))
        return {"bundle": tmp_path / "bundle.json", "comfyui": tmp_path / "comfyui.json", "canva": tmp_path / "canva.json"}

    monkeypatch.setattr(generate_thumbnail, "generate_move_church_thumbnail", fake_generate)
    monkeypatch.setattr(generate_thumbnail, "export_thumbnail_studio_bundle", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_thumbnail.py",
            "--source",
            "/tmp/source.mp4",
            "--title",
            "GOTT IST TREU",
            "--template",
            "warm_gold",
            "--output-dir",
            str(tmp_path),
            "--export-studio-bundle",
            "--studio-notes",
            "Canva review handoff",
        ],
    )

    generate_thumbnail._cli()

    assert generate_calls[0][1]["output_path"] == str(tmp_path / "source_warm_gold_9x16.png")
    assert export_calls[0][0] == str(tmp_path / "source_warm_gold_9x16.png")
    assert export_calls[0][1]["notes"] == "Canva review handoff"


def test_invalid_hook_rejects_visual_descriptions():
    assert _is_invalid_hook(
        "Ein verwirrter Gesichtsausdruck mit einem Smartphone",
        "Smartphone",
        clip_transcript="verwirrt gesichtsausdruck smartphone",
        language="de",
    )


def test_background_prompt_mentions_sermon_background_and_excludes_visual_noise():
    positive, negative = build_background_prompt(
        "Standhaft in der Endzeit",
        template="bold_minimal",
        speaker_name="Antonio Weil",
        brand_label="Move Church",
    )

    assert "cinematic sermon background" in positive
    assert "negative space" in positive
    assert "no people" in positive
    assert "smartphone" in negative
    assert "watermark" in negative


def test_prepare_workflow_overrides_dimensions_prompt_seed_and_prefix():
    workflow = {
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old positive"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "old negative"}},
        "13": {"class_type": "SamplerCustom", "inputs": {"noise_seed": 0}},
        "20": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"}},
        "22": {"class_type": "SDTurboScheduler", "inputs": {"steps": 1, "denoise": 1}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI"}},
    }

    prepared = _prepare_workflow(
        workflow,
        prompt="cinematic sermon background",
        negative_prompt="people, text, watermark",
        width=832,
        height=1472,
        seed=123456,
        filename_prefix="parakeet_bg_test_123456",
    )

    assert prepared["5"]["inputs"]["width"] == 832
    assert prepared["5"]["inputs"]["height"] == 1472
    assert prepared["6"]["inputs"]["text"] == "cinematic sermon background"
    assert prepared["7"]["inputs"]["text"] == "people, text, watermark"
    assert prepared["13"]["inputs"]["noise_seed"] == 123456
    assert prepared["27"]["inputs"]["filename_prefix"] == "parakeet_bg_test_123456"
