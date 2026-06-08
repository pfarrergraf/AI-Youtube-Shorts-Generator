from __future__ import annotations

import io

from PIL import Image

import Components.ComfyUIBackground as cb


def _make_png(path, color=(40, 60, 90)):
    Image.new("RGB", (320, 480), color).save(path, "PNG")
    return path


def _png_bytes(color=(200, 120, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (320, 480), color).save(buffer, "PNG")
    return buffer.getvalue()


def test_build_img2img_workflow_wires_input_and_prompt():
    workflow = cb._build_img2img_workflow(
        prompt="warm cinematic grade",
        negative_prompt="text, watermark",
        input_filename="frame.png",
        denoise=0.55,
        steps=8,
        cfg=1.0,
        seed=123,
        checkpoint="sd_xl_turbo_1.0_fp16.safetensors",
        filename_prefix="parakeet_edit_123",
    )
    assert workflow["load"]["inputs"]["image"] == "frame.png"
    assert workflow["pos"]["inputs"]["text"] == "warm cinematic grade"
    assert workflow["neg"]["inputs"]["text"] == "text, watermark"
    assert workflow["samp"]["inputs"]["denoise"] == 0.55
    assert workflow["samp"]["inputs"]["seed"] == 123
    assert workflow["enc"]["inputs"]["pixels"] == ["load", 0]
    assert workflow["save"]["inputs"]["filename_prefix"] == "parakeet_edit_123"


def test_build_background_prompt_adds_scene_hint_for_endzeit():
    prompt, negative = cb.build_background_prompt(
        "Standhaft in der Endzeit",
        template="bold_minimal",
        speaker_name="Antonio Weil",
        brand_label="Move Church",
    )

    lowered = prompt.lower()
    assert "stormy horizon" in lowered
    assert "broken stone" in lowered
    assert "antonio weil" in lowered
    assert "move church" in lowered
    assert "no people" in lowered
    assert "text" in negative.lower()


def test_generate_edited_image_round_trip(monkeypatch, tmp_path):
    src = _make_png(tmp_path / "frame.png")
    captured = {}

    monkeypatch.setattr(cb, "_upload_image", lambda path, base_url=cb._DEFAULT_COMFYUI_URL: "frame.png")

    def fake_queue(prompt, base_url=cb._DEFAULT_COMFYUI_URL):
        captured["prompt"] = prompt
        return "edit-1"

    monkeypatch.setattr(cb, "_queue_prompt", fake_queue)
    monkeypatch.setattr(
        cb,
        "_fetch_history",
        lambda prompt_id, base_url=cb._DEFAULT_COMFYUI_URL: {
            "edit-1": {"outputs": {"save": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}
        },
    )
    monkeypatch.setattr(cb, "_download_image", lambda ref, base_url=cb._DEFAULT_COMFYUI_URL: _png_bytes())

    image, info = cb.generate_edited_image(
        input_image=src,
        prompt="warm cinematic grade",
        denoise=0.55,
        steps=8,
        cache_dir=tmp_path / "cache",
    )

    assert info["backend"] == "comfyui"
    assert info["prompt_id"] == "edit-1"
    assert captured["prompt"]["load"]["inputs"]["image"] == "frame.png"
    assert image.size == (320, 480)
    cache_path = info["cache_path"]
    assert cache_path and cache_path.endswith(".png")

    # Second call hits the cache and skips the server entirely.
    monkeypatch.setattr(cb, "_upload_image", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not upload")))
    cached_image, cached_info = cb.generate_edited_image(
        input_image=src,
        prompt="warm cinematic grade",
        denoise=0.55,
        steps=8,
        cache_dir=tmp_path / "cache",
    )
    assert cached_info["backend"] == "cache"
    assert cached_image.size == (320, 480)


def test_generate_edited_image_falls_back_to_source_on_error(monkeypatch, tmp_path):
    src = _make_png(tmp_path / "frame.png", color=(10, 20, 30))
    monkeypatch.setattr(cb, "_upload_image", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("server down")))

    image, info = cb.generate_edited_image(input_image=src, prompt="grade")

    assert info["backend"] == "fallback"
    assert "server down" in info["reason"]
    assert image.size == (320, 480)


def test_generate_background_image_uses_explicit_prompt_override(monkeypatch, tmp_path):
    workflow = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old positive"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "old negative"}},
        "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1472, "batch_size": 1}},
        "4": {"class_type": "KSampler", "inputs": {"seed": 1}},
        "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "old"}},
    }
    captured = {}

    monkeypatch.setattr(cb, "_load_workflow_template", lambda path: workflow)
    monkeypatch.setattr(cb, "_workflow_path", lambda comfyui_root, workflow_name: tmp_path / "workflow.json")

    def _fake_queue(prompt, base_url=cb._DEFAULT_COMFYUI_URL):
        captured["prompt"] = prompt
        return "bg-1"

    monkeypatch.setattr(cb, "_queue_prompt", _fake_queue)
    monkeypatch.setattr(
        cb,
        "_fetch_history",
        lambda prompt_id, base_url=cb._DEFAULT_COMFYUI_URL: {
            "bg-1": {"outputs": {"5": {"images": [{"filename": "bg.png", "subfolder": "", "type": "output"}]}}}
        },
    )
    monkeypatch.setattr(cb, "_download_image", lambda ref, base_url=cb._DEFAULT_COMFYUI_URL: _png_bytes())

    image, info = cb.generate_background_image(
        title="ignored",
        prompt="cinematic sermon background, stormy horizon",
        negative_prompt="text, watermark",
        cache_dir=tmp_path / "cache",
    )

    assert info["backend"] == "comfyui"
    assert captured["prompt"]["1"]["inputs"]["text"] == "cinematic sermon background, stormy horizon"
    assert captured["prompt"]["2"]["inputs"]["text"] == "text, watermark"
    assert image.size == (832, 1472)
