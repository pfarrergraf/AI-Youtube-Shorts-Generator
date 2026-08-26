"""
Offline test for the HIGHLIGHT_SYSTEM_PROMPT_FILE override in GetAllHighlights.

Nothing here needs a GPU or a real LLM: _call_llm is monkeypatched to capture
the system prompt it was handed and return a single well-formed candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Components import LanguageTasks as lt  # noqa: E402

TRANSCRIPT = "\n".join(
    f"[{i:.1f}s] Satz Nummer {i} der Predigt geht so weiter und weiter."
    for i in range(0, 40, 2)
)

CANDIDATE = {
    "start": 4.0,
    "end": 34.0,
    "content": "Ein vollstaendiger Gedanke.",
    "title": "Beispiel",
    "hook": "Beispiel-Hook",
    "impact": 8,
    "confidence": 0.9,
    "why": "klarer Aufbau",
    "opening_complete": True,
    "ending_complete": True,
    "cliffhanger": False,
    "payoff_excerpt": "Satz Nummer 30",
}


def _patch_call_llm(monkeypatch, captured):
    def fake_call_llm(system_prompt, user_content, temperature=0.5, _retries=3):
        captured.append(system_prompt)
        return json.dumps([CANDIDATE])

    monkeypatch.setattr(lt, "_call_llm", fake_call_llm)


def test_default_prompt_used_when_env_unset(monkeypatch):
    monkeypatch.delenv("HIGHLIGHT_SYSTEM_PROMPT_FILE", raising=False)
    captured = []
    _patch_call_llm(monkeypatch, captured)

    highlights = lt.GetAllHighlights(TRANSCRIPT)

    assert captured == [lt.MULTI_HIGHLIGHT_PROMPT]
    assert len(highlights) == 1


def test_override_file_used_when_env_set(monkeypatch, tmp_path):
    override = tmp_path / "custom_highlight_prompt.txt"
    override.write_text("CUSTOM SYSTEM PROMPT FOR KERSTIN GRAF LANE", encoding="utf-8")
    monkeypatch.setenv("HIGHLIGHT_SYSTEM_PROMPT_FILE", str(override))
    captured = []
    _patch_call_llm(monkeypatch, captured)

    highlights = lt.GetAllHighlights(TRANSCRIPT)

    assert captured == ["CUSTOM SYSTEM PROMPT FOR KERSTIN GRAF LANE"]
    assert len(highlights) == 1


def test_missing_override_file_falls_back_to_default(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    monkeypatch.setenv("HIGHLIGHT_SYSTEM_PROMPT_FILE", str(missing))
    captured = []
    _patch_call_llm(monkeypatch, captured)

    highlights = lt.GetAllHighlights(TRANSCRIPT)

    assert captured == [lt.MULTI_HIGHLIGHT_PROMPT]
    assert len(highlights) == 1
