from __future__ import annotations

from PIL import Image

from Components.ThumbnailStoryAssets import add_foreground_assets, resolve_story_assets


def test_approved_story_background_resolves_to_requested_canvas():
    layers = resolve_story_assets(["daniel_lions_den"], (540, 960))
    assert layers["background_id"] == "daniel_lions_den"
    assert layers["background"].size == (540, 960)
    assert layers["speaker_anchor_x"] == 0.76
    assert layers["story_focus_box"] == [10, 134, 334, 595]
    assert layers["rejected"] == []


def test_unknown_story_asset_is_rejected_without_breaking_render():
    layers = resolve_story_assets(["does_not_exist"], (540, 960))
    assert layers["background"] is None
    assert layers["rejected"] == ["does_not_exist"]


def test_foreground_overlay_receives_shadow_and_is_reported():
    layers = resolve_story_assets(["oil_lamp_darkness"], (540, 960))
    canvas = Image.new("RGB", (540, 960), "#101522")
    rendered, report = add_foreground_assets(
        canvas, layers["foreground"], light_direction="upper_left"
    )
    assert rendered.size == canvas.size
    assert report[0]["asset_id"] == "oil_lamp_darkness"
    assert rendered.getbbox() is not None
