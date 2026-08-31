from __future__ import annotations

import json

from Components import LanguageTasks as language_tasks


def test_thumbnail_brief_returns_story_pose_layers_and_social_title(monkeypatch):
    monkeypatch.setattr(
        language_tasks,
        "GenerateTitleHook",
        lambda *args, **kwargs: ("DEIN GOLIATH FÄLLT", "GOLIATH"),
    )
    response = {
        "hook_text": "DEIN GOLIATH FÄLLT",
        "accent_keyword": "GOLIATH",
        "social_title": "Mit Gott überwinden",
        "story_concept": "David faces Goliath while the preacher observes from safety.",
        "curiosity_gap": "How can David possibly win?",
        "emotion_target": "courage",
        "pose_id": "battle_ready",
        "story_asset_ids": ["david_goliath", "not_in_catalog"],
        "speaker_side": "right",
        "background_style": "strong_contrast",
        "background_prompt": "dusty valley at sunset, empty foreground",
        "background_negative_prompt": "text, logo, watermark",
        "layer_plan": ["background_plate", "story_midground", "speaker", "foreground_eye_catcher", "final_grade"],
        "light_direction": "sunset_right",
        "palette": ["sand gold", "storm blue", "blood red"],
        "art_direction": "baroque_drama",
        "brand_label": "MOVE CHURCH",
    }
    monkeypatch.setattr(language_tasks, "_call_llm", lambda *args, **kwargs: json.dumps(response))

    brief = language_tasks.GenerateThumbnailBrief(
        "David trusts God against Goliath",
        clip_transcript="David vertraute Gott: Dein Goliath fällt.",
        video_title="Mit Gott überwinden",
        language="de",
        speaker_name="Thomas Herrmann",
        brand_label="MOVE CHURCH",
    )

    assert brief["social_title"].startswith(brief["hook_text"])
    assert brief["pose_id"] == "battle_ready"
    assert brief["story_asset_ids"] == ["david_goliath"]
    assert brief["layer_plan"].index("speaker") < brief["layer_plan"].index("final_grade")
    assert brief["palette"] == ["sand gold", "storm blue", "blood red"]
    assert brief["art_direction"] == "baroque_drama"


def test_thumbnail_brief_normalises_cyrillic_homoglyph_in_german_hook(monkeypatch):
    monkeypatch.setattr(
        language_tasks,
        "GenerateTitleHook",
        lambda *args, **kwargs: ("DER STURM IST REAL", "STURM"),
    )
    response = {
        "hook_text": "JESUS IST ENTSPANNТ",  # final character is Cyrillic U+0422
        "accent_keyword": "ENTSPANNТ",
        "social_title": "JESUS IST ENTSPANNТ",
    }
    monkeypatch.setattr(language_tasks, "_call_llm", lambda *args, **kwargs: json.dumps(response))

    brief = language_tasks.GenerateThumbnailBrief(
        "Jesus bleibt im Sturm ruhig",
        clip_transcript="Jesus ist entspannt, obwohl der Sturm real ist.",
        video_title="Jesus im Boot",
        language="de",
    )

    assert brief["hook_text"] == "JESUS IST ENTSPANNT"
    assert brief["accent_keyword"] == "ENTSPANNT"
    assert brief["social_title"].startswith("JESUS IST ENTSPANNT")
    assert all("LATIN" in language_tasks.unicodedata.name(ch, "") for ch in brief["hook_text"] if ch.isalpha())


def test_hook_validator_rejects_unmapped_non_latin_letters():
    assert language_tasks._is_invalid_hook(
        "JESUS IST ENTSPАNNTЖ",
        "ENTSPАNNTЖ",
        clip_transcript="Jesus ist entspannt.",
        language="de",
    )
