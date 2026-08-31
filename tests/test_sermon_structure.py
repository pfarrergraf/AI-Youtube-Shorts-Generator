"""Offline contract tests for structure-first sermon editing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Components import LanguageTasks as lt  # noqa: E402


TRANSCRIPT = "[0.0s] These.\n[8.0s] Beispiel.\n[18.0s] Deshalb ist Gott im Boot."


def test_structure_map_requires_coherent_timestamped_blocks(monkeypatch):
    payload = {
        "overall_thesis": "Gott ist im Sturm gegenwärtig.",
        "blocks": [
            {"start": 0, "end": 8, "role": "thesis", "summary": "These", "payoff": ""},
            {"start": 8, "end": 18, "role": "example", "summary": "Beispiel", "payoff": "Gott im Boot"},
            {"start": 20, "end": 20, "role": "argument", "summary": "invalid"},
        ],
    }
    monkeypatch.setattr(lt, "_call_llm", lambda *_args, **_kwargs: json.dumps(payload))

    structure = lt.AnalyseSermonStructure(TRANSCRIPT)

    assert structure is not None
    assert structure["overall_thesis"] == "Gott ist im Sturm gegenwärtig."
    assert [(block["start"], block["end"]) for block in structure["blocks"]] == [(0.0, 8.0), (8.0, 18.0)]


def test_highlight_prompt_receives_structure_as_boundary_context(monkeypatch):
    captured = []
    structure = {
        "overall_thesis": "Gott ist im Sturm gegenwärtig.",
        "blocks": [{"start": 0.0, "end": 18.0, "role": "argument", "summary": "vollständiger Bogen"}],
    }
    candidate = {
        "start": 0.0, "end": 18.0, "title": "Gott im Boot", "hook": "Nicht allein",
        "content": "Vollständiger Gedanke", "impact": 8, "confidence": 0.9,
        "opening_complete": True, "ending_complete": True, "cliffhanger": False,
        "payoff_excerpt": "Gott ist im Boot.", "why": "Bogen", "transcript_excerpt": "These Beispiel Pointe",
    }

    def fake_call(_prompt, content, **_kwargs):
        captured.append(content)
        return json.dumps([candidate])

    monkeypatch.setattr(lt, "_call_llm", fake_call)
    highlights = lt.GetAllHighlights(TRANSCRIPT, sermon_structure=structure)

    assert len(highlights) == 1
    assert "Precomputed sermon structure" in captured[0]
    assert "Gott ist im Sturm gegenwärtig." in captured[0]
