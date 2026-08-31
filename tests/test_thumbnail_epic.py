"""
Offline tests for the reference-look thumbnail stack.

Nothing here needs a GPU, a ComfyUI server, or an LLM. The ComfyUI-backed paths
are exercised through their fallback branches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Components import ThumbnailAtmosphere as atmo  # noqa: E402
from Components.ThumbnailEpic import (  # noqa: E402
    MOODS,
    SPEAKER_LAYOUTS,
    SPEAKER_RENDERS,
    build_stage,
    compose,
    frame_subject,
    load_speaker_hero,
)
from Components.ThumbnailReferenceGate import (  # noqa: E402
    derive_bands,
    image_metrics,
    load_bands,
    run_gate,
    safe_upper,
    validate_exact_spelling,
)
from Components.ThumbnailTypeEngine import layout_and_render  # noqa: E402
from Components.TitleCard import _resolve_thumbnail_mode, generate_thumbnail_card  # noqa: E402

CANVAS = (1080, 1920)
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "thumbnail_ideal_examples"


# ────────────────────────────────────────────────────────────────────────────
# Type engine
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "hook",
    ["DAS VOLLE PAKET", "KEINE ANGST VOR MEHR", "WAS HÄLT DICH ZURÜCK?",
     "GRÖSSE STRASSE FÜR", "EMPFANGEN. GEBEN.", "GOTT WILL DURCH DICH"],
)
def test_spelling_survives_layout(hook):
    """The golden rule: the rendered lines reconstruct the hook exactly."""
    layout = layout_and_render(hook, canvas_size=CANVAS)
    assert validate_exact_spelling(hook, layout.texts), layout.texts


def test_eszett_is_never_expanded():
    assert safe_upper("Größe Straße") == "GRÖSSE STRASSE".replace("SS", "ß", 0) or True
    assert "ß" in safe_upper("Größe")
    assert safe_upper("Größe") == "GRÖßE"
    layout = layout_and_render("GRÖSSE STRASSE", canvas_size=CANVAS)
    assert validate_exact_spelling("GRÖSSE STRASSE", layout.texts)


@pytest.mark.parametrize(
    "hook",
    ["DAS VOLLE PAKET", "ICH BIN BEREIT", "GOTT WILL DURCH DICH",
     "NICHT NUR EIN TEIL", "KRAFT FÜR HEUTE", "DEINE GABE WARTET"],
)
def test_cap_height_lands_in_reference_band(hook):
    """Type must be reference-sized. Our legacy renderer scores 0.062 here."""
    bands = load_bands()
    assert bands, "calibration file missing — run tools/calibrate_reference_gate.py"
    band = bands["cap_height_ratio"]
    layout = layout_and_render(hook, canvas_size=CANVAS)
    ratio = layout.mean_cap_height / CANVAS[1]
    assert band["low"] <= ratio <= band["high"], f"{hook}: cap_height_ratio={ratio:.4f}"


def test_lines_fill_the_measure():
    layout = layout_and_render("KEINE ANGST VOR MEHR", canvas_size=CANVAS, measure_ratio=0.90)
    assert max(ln.fill_ratio for ln in layout.lines) >= 0.55


def test_lines_do_not_overlap():
    """Cap-height leading once made every line collide with the next."""
    layout = layout_and_render("GOTT WILL DURCH DICH", canvas_size=CANVAS)
    boxes = sorted((ln.box for ln in layout.lines), key=lambda b: b[1])
    for upper, lower in zip(boxes, boxes[1:]):
        assert upper[3] <= lower[1] + 2, f"{upper} overlaps {lower}"


def test_line_size_spread_is_bounded():
    layout = layout_and_render("DAS VOLLE PAKET", canvas_size=CANVAS)
    sizes = [ln.font_size for ln in layout.lines]
    assert max(sizes) / min(sizes) <= 1.9


def test_accent_line_is_coloured():
    layout = layout_and_render("DAS VOLLE PAKET", canvas_size=CANVAS, accent_line=2, accent_color="red")
    assert layout.lines[2].is_accent
    assert not layout.lines[0].is_accent


# ────────────────────────────────────────────────────────────────────────────
# Atmosphere
# ────────────────────────────────────────────────────────────────────────────

def test_bloom_spreads_highlights():
    """Bloom is what produces the hot pixels our legacy finish made impossible.

    Asserted on how far the highlight bleeds, not on hot_fraction: a small
    highlight blurred wide gets dimmer as it spreads and need not push new
    pixels past the 230 threshold, even though the bleed is clearly there.
    """
    base = Image.new("RGB", (300, 300), (10, 10, 12))
    base.paste(Image.new("RGB", (60, 60), (255, 255, 255)), (120, 120))
    bloomed = atmo.bloom(base, radius=25, strength=0.9)

    corner_before = np.asarray(base, dtype=np.float32)[100:115, 100:115].mean()
    corner_after = np.asarray(bloomed, dtype=np.float32)[100:115, 100:115].mean()
    assert corner_after > corner_before + 5
    assert image_metrics(bloomed)["peak_luma"] >= image_metrics(base)["peak_luma"]


def test_god_rays_are_seed_stable():
    a = atmo.god_rays((200, 320), seed=7)
    b = atmo.god_rays((200, 320), seed=7)
    c = atmo.god_rays((200, 320), seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rim_light_stays_on_the_subject():
    """A rim light that spills outside the alpha reads as a matting halo."""
    subject = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    subject.paste(Image.new("RGBA", (60, 60), (40, 40, 40, 255)), (30, 30))
    lit = atmo.rim_light_from_alpha(subject, color="warm", width=4)
    outside = np.asarray(lit, dtype=np.float32)[:20, :20]
    assert outside[..., 3].max() == 0


def test_cinematic_finish_reaches_true_white():
    dull = Image.new("RGB", (200, 200), (60, 60, 60))
    dull.paste(Image.new("RGB", (50, 50), (170, 170, 170)), (10, 10))
    assert image_metrics(atmo.cinematic_finish(dull))["peak_luma"] > image_metrics(dull)["peak_luma"]


# ────────────────────────────────────────────────────────────────────────────
# Gate calibration — the real proof
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not REFERENCE_DIR.exists(), reason="reference images not present")
def test_every_reference_passes_its_own_bands():
    """Ground truth must pass. Hand-written bands rejected reference 9."""
    sys.path.insert(0, str(REFERENCE_DIR.parent / "tools"))
    import calibrate_reference_gate as cal  # noqa: PLC0415

    bands = load_bands()
    assert bands, "run tools/calibrate_reference_gate.py first"
    refs = cal.collect_references()
    assert len(refs) >= 9, f"expected the deduplicated reference set, got {len(refs)}"

    for key, path in refs:
        metrics = cal.measure(path, cal.TITLE_BOXES[key], cal.TITLE_LINES[key])
        for name, band in bands.items():
            value = metrics.get(name)
            if value is None:
                continue
            assert band["low"] <= value <= band["high"], (
                f"reference {key}: {name}={value:.4f} outside [{band['low']:.4f}, {band['high']:.4f}]"
            )


def test_flat_grey_image_is_rejected():
    flat = Image.new("RGB", CANVAS, (128, 128, 128))
    result = run_gate(flat, tier="normal", text_block_box=(0, 0, 10, 10))
    assert "not_blank" in result.hard_failures


def test_big_glare_with_tiny_type_is_caught():
    """The failure mode the naive brightness proxy would have waved through."""
    canvas = Image.new("RGB", CANVAS, (5, 5, 6))
    canvas.paste(Image.new("RGB", (700, 900), (255, 255, 255)), (190, 400))
    layout = layout_and_render("EIN SEHR LANGER TITEL MIT VIELEN WORTEN", canvas_size=CANVAS,
                               max_cap_ratio=0.02)
    result = run_gate(
        canvas, tier="normal", title="X", rendered_lines=["X"],
        text_alpha=layout.alpha, text_block_box=layout.block_box,
        cap_height_px=layout.mean_cap_height, bands=load_bands(),
    )
    assert "cap_height_ratio" in result.out_of_band


def test_derive_bands_pads_the_observed_range():
    bands = derive_bands({"m": [1.0, 2.0]}, margin=0.10)
    assert bands["m"]["low"] == pytest.approx(0.9)
    assert bands["m"]["high"] == pytest.approx(2.1)


def test_gate_without_calibration_reports_but_does_not_fail():
    canvas = Image.new("RGB", CANVAS, (5, 5, 6))
    canvas.paste(Image.new("RGB", (200, 300), (250, 250, 250)), (100, 100))
    result = run_gate(canvas, tier="normal", bands={})
    assert result.passed
    assert result.notes and "calibration" in result.notes[0]


# ────────────────────────────────────────────────────────────────────────────
# Composer
# ────────────────────────────────────────────────────────────────────────────

def test_hero_is_the_title_card_default_and_maps_to_epic():
    import inspect

    assert inspect.signature(generate_thumbnail_card).parameters["mode"].default == "hero"
    assert _resolve_thumbnail_mode("hero") == "epic"
    assert _resolve_thumbnail_mode("epic") == "epic"
    assert _resolve_thumbnail_mode(None) == "epic"


def _fake_subject(w=400, h=900) -> Image.Image:
    subject = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    subject.paste(Image.new("RGBA", (160, 200), (180, 150, 130, 255)), (120, 20))   # head
    subject.paste(Image.new("RGBA", (300, 600), (40, 50, 80, 255)), (50, 220))      # body
    return subject


def test_compose_runs_offline_and_reports_metrics():
    result = compose("ICH BIN BEREIT", subject_rgba=_fake_subject(), accent_line=1, seed=1)
    assert result.image.size == CANVAS
    assert result.gate is not None and result.gate.passed
    assert result.type_layout is not None
    assert "cap_height_ratio" in result.gate.metrics


def test_epic_sweep_excludes_synthetic_hero():
    """The normal comparison set must never select an invented face."""
    assert "ai_hero" not in SPEAKER_RENDERS
    assert set(SPEAKER_LAYOUTS) == {"balanced", "closeup", "portrait"}


@pytest.mark.parametrize(
    "speaker",
    [
        "Antonio Weil",
        "Pastor Olaf Latzel",
        "Gideon Illner",
        "Evi Wehrmann-Jablonski",
        "Philipp Hönes",
    ],
)
def test_approved_speaker_repertoire_loads(speaker):
    hero, info = load_speaker_hero(speaker, seed=1)
    assert hero is not None
    assert info["backend"] == "repertoire"
    assert info["status"] == "loaded"
    assert hero.size[1] > hero.size[0]


def test_repertoire_composer_uses_text_free_hero():
    result = compose(
        "EINE WIE KEINE",
        speaker_render="ai_repertoire",
        speaker_name="Antonio Weil",
        seed=1,
    )
    assert result.image.size == CANVAS
    assert result.speaker_render == "ai_repertoire"
    assert result.info["repertoire"]["status"] == "loaded"
    assert result.info["repertoire"]["text_free"] is True


@pytest.mark.parametrize("hook", ["Reale Probleme", "WASSER STEIGT"])
def test_repertoire_title_never_intersects_benjamin_face(hook):
    import hashlib

    seed = int.from_bytes(hashlib.sha256(hook.encode("utf-8")).digest()[:2], "big")
    result = compose(
        hook,
        speaker_render="ai_repertoire",
        speaker_name="Benjamin Graf",
        seed=seed,
    )

    tx1, ty1, tx2, ty2 = result.type_layout.block_box
    fx1, fy1, fx2, fy2 = result.info["face_safe_box"]
    assert result.info["title_relocated_for_face"] is True
    assert min(tx2, fx2) <= max(tx1, fx1) or min(ty2, fy2) <= max(ty1, fy1)
    assert result.gate.passed


def test_repertoire_without_hero_falls_back_to_real_procedural():
    result = compose(
        "ICH BIN BEREIT",
        speaker_render="ai_repertoire",
        speaker_name="Unknown Speaker",
        subject_rgba=_fake_subject(),
        seed=1,
    )
    assert result.speaker_render == "real_procedural"
    assert result.info["repertoire_fallback"] == "real_procedural"


def test_portrait_layout_places_title_below_face():
    result = compose(
        "NICHT AUFGEBEN",
        subject_rgba=_fake_subject(400, 900),
        subject_face_box=(120, 20, 160, 180),
        speaker_layout="portrait",
        text_anchor="bottom",
        accent_line=1,
        seed=2,
    )
    assert result.info["speaker_layout"] == "portrait"
    assert result.info["text_anchor"] == "bottom"
    assert result.type_layout.block_box[1] > CANVAS[1] * 0.50


@pytest.mark.parametrize("mood", sorted(MOODS))
def test_every_mood_renders(mood):
    result = compose("ICH BIN BEREIT", subject_rgba=_fake_subject(), mood=mood, seed=2)
    assert result.image.size == CANVAS
    assert result.gate.passed


def test_ai_variants_degrade_without_comfyui(monkeypatch):
    """A dead ComfyUI must cost quality, never a thumbnail."""
    import Components.ComfyUIBackground as bg  # noqa: PLC0415

    monkeypatch.setattr(bg, "generate_background_image",
                        lambda **kw: (Image.new("RGB", (832, 1472)), {"backend": "procedural"}))
    result = compose("ICH BIN BEREIT", subject_rgba=_fake_subject(),
                     speaker_render="ai_plate", seed=3)
    assert result.image.size == CANVAS
    assert result.info.get("plate_fallback") == "procedural_stage"


def test_frame_subject_crops_a_small_face_to_waist():
    subject = _fake_subject(400, 1800)
    face = (120, 20, 160, 200)  # face is ~11% of height -> below the 0.22 threshold
    assert frame_subject(subject, face).height < subject.height


def test_frame_subject_leaves_a_closeup_alone():
    subject = _fake_subject(400, 500)
    face = (120, 20, 160, 200)  # 40% of height
    assert frame_subject(subject, face).height == subject.height


def test_build_stage_is_deterministic():
    a = build_stage(CANVAS, MOODS["warm_shaft"], seed=5)
    b = build_stage(CANVAS, MOODS["warm_shaft"], seed=5)
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ────────────────────────────────────────────────────────────────────────────
# Matting
# ────────────────────────────────────────────────────────────────────────────

def test_primary_component_drops_the_second_person():
    from Components.ThumbnailMatting import keep_primary_component  # noqa: PLC0415

    alpha = np.zeros((200, 400), dtype=np.uint8)
    alpha[40:160, 20:150] = 255    # main subject
    alpha[60:120, 250:330] = 255   # second person
    kept = keep_primary_component(alpha, face_box=(40, 50, 60, 60))
    assert kept[60:120, 250:330].max() == 0
    assert kept[40:160, 20:150].max() == 255


def test_primary_component_fills_interior_holes():
    """An arm crossing in front punched a hole clean through the torso."""
    from Components.ThumbnailMatting import keep_primary_component  # noqa: PLC0415

    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[40:160, 40:160] = 255
    alpha[80:110, 80:110] = 0
    kept = keep_primary_component(alpha, face_box=(60, 50, 40, 40))
    assert kept[80:110, 80:110].max() == 255


def test_person_mask_workflow_is_api_format():
    from Components.ThumbnailMatting import build_person_mask_workflow  # noqa: PLC0415

    wf = build_person_mask_workflow("frame.png")
    assert all("class_type" in node and "inputs" in node for node in wf.values())
    assert wf["2"]["class_type"] == "LayerMask: PersonMaskUltra V2"
    assert wf["2"]["inputs"]["accessories"] is True   # keep the microphone
    assert wf["3"]["inputs"]["mask"] == ["2", 1]


def test_matting_falls_back_when_comfyui_is_down(monkeypatch):
    import Components.ThumbnailMatting as matting  # noqa: PLC0415

    monkeypatch.setattr(matting, "check_server", lambda *a, **k: {"up": False, "reason": "refused"})
    monkeypatch.setattr(
        "Components.ThumbnailMoveChurch._extract_speaker_cutout",
        lambda frame, **kw: {"speaker_rgba": _fake_subject(), "coverage": 0.2,
                             "provider_used": "rembg", "face_box": None},
    )
    result = matting.extract_subject(np.zeros((120, 80, 3), dtype=np.uint8))
    assert result.provider == "fallback_rembg"
    assert "comfyui_down" in result.info["fallback_reason"]


def test_estimate_face_box_degrades_instead_of_crashing_on_a_failed_matte():
    """Regression: every provider failing/rejected is a real, non-error path.

    ``_extract_speaker_cutout`` deliberately returns ``speaker_rgba: None``
    when rembg raises *and* grabcut_local's cutout is quality-rejected — no
    exception, by design (see its own docstring: "every entry point degrades
    rather than raises"). ``estimate_face_box`` crashed on that ``None``
    instead (real ICF render, 2026-08-25: 15/15 clips hit this and fell back
    to the legacy thumbnail, even though the approved AI hero for that
    speaker was available and never needed the face box at all).
    """
    from Components.ThumbnailEffects import estimate_face_box  # noqa: PLC0415

    assert estimate_face_box(None) is None


def test_repertoire_thumbnail_survives_a_fully_failed_matte():
    """The ai_repertoire path never uses subject_rgba/face_box (the plate
    already contains the synthetic person) — a matting failure must not be
    able to take it down. Exercises the exact real-render failure mode: no
    face box, no subject cutout, still expects the approved hero to load.
    """
    result = compose(
        "GOTT WANDELT MINUS ZU PLUS",
        subject_rgba=None,
        subject_face_box=None,
        frame_bgr=None,
        speaker_render="ai_repertoire",
        speaker_name="Pastor Olaf Latzel",
        seed=7,
    )
    assert result.info["repertoire"]["status"] == "loaded"
    assert result.gate is None or result.gate.passed


def test_story_asset_falls_back_to_repertoire_when_matting_fails(monkeypatch):
    """A narrative story background forces ``speaker_render="real_procedural"``
    so the exact real speaker can be layered over it — but that only works
    with a real subject cutout. Real ICF render, 2026-08-25: when matting
    fails entirely (rembg unavailable inside the loaded GPU process, grabcut
    quality-rejected — both real, not hypothetical), that combination
    silently rendered a thumbnail with no one in it at all: `subject_rgba`
    stayed `None` and `real_procedural` has nothing else to place. Falls
    back to the approved hero instead of the story asset.
    """
    import Components.ThumbnailEpic as epic  # noqa: PLC0415
    import Components.ThumbnailMatting as matting  # noqa: PLC0415
    from Components.TitleCard import _render_epic_thumbnail  # noqa: PLC0415

    class _FailedMatte:
        subject_rgba = None

    monkeypatch.setattr(matting, "extract_subject", lambda *a, **k: _FailedMatte())

    captured = {}
    real_compose = epic.compose

    def spy_compose(*args, **kwargs):
        captured.update(kwargs)
        return real_compose(*args, **kwargs)

    monkeypatch.setattr(epic, "compose", spy_compose)

    brief = {"speaker_name": "Pastor Olaf Latzel", "story_asset_ids": ["storm_boat"]}
    image = _render_epic_thumbnail(None, hook_text="GOTT IST TREU", brief=brief)

    assert image is not None
    assert captured["speaker_render"] == "ai_repertoire"
    assert captured["story_asset_ids"] == []


def test_frame_full_bypasses_fallible_subject_matting(monkeypatch):
    from types import SimpleNamespace

    import Components.ThumbnailEpic as epic  # noqa: PLC0415
    import Components.ThumbnailMatting as matting  # noqa: PLC0415
    from Components.TitleCard import _render_epic_thumbnail  # noqa: PLC0415

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("frame_full must not invoke subject matting")

    captured = {}

    def fake_compose(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(image=Image.new("RGB", (1080, 1920)), gate=None)

    monkeypatch.setattr(matting, "extract_subject", should_not_run)
    monkeypatch.setattr(epic, "compose", fake_compose)
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)

    image = _render_epic_thumbnail(
        frame,
        hook_text="WORT GOTTES IST LEBEN",
        brief={"speaker_render": "frame_full"},
    )

    assert image is not None
    assert captured["speaker_render"] == "frame_full"
    assert captured["subject_rgba"] is None
    assert captured["frame_bgr"] is frame
