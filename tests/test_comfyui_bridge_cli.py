from __future__ import annotations

import json

from PIL import Image

import comfyui_bridge_cli as cli


def _png(path, color=(30, 40, 60)):
    Image.new("RGB", (64, 96), color).save(path, "PNG")
    return path


def test_status_returns_nonzero_when_down(monkeypatch, capsys):
    monkeypatch.setattr(cli, "check_server", lambda base_url: {"up": False, "reason": "boom"})
    rc = cli.main(["status"])
    assert rc == 1
    assert "boom" in capsys.readouterr().out


def test_background_wires_args_and_saves(monkeypatch, tmp_path):
    captured = {}

    def fake_bg(**kwargs):
        captured.update(kwargs)
        return Image.new("RGBA", (64, 96), (10, 20, 30, 255)), {"backend": "comfyui", "seed": 7}

    monkeypatch.setattr(cli, "generate_background_image", fake_bg)
    out = tmp_path / "bg.png"
    rc = cli.main(["background", "--title", "Gnade", "--template", "warm_gold", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert captured["title"] == "Gnade"
    assert captured["template"] == "warm_gold"


def test_edit_wires_args_and_saves(monkeypatch, tmp_path):
    src = _png(tmp_path / "frame.png")
    captured = {}

    def fake_edit(**kwargs):
        captured.update(kwargs)
        return Image.new("RGBA", (64, 96), (50, 50, 50, 255)), {"backend": "comfyui"}

    monkeypatch.setattr(cli, "generate_edited_image", fake_edit)
    out = tmp_path / "edited.png"
    rc = cli.main(["edit", "--input", str(src), "--prompt", "warm grade", "--denoise", "0.5", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert captured["denoise"] == 0.5
    assert captured["prompt"] == "warm grade"


def test_run_rejects_ui_format(monkeypatch, tmp_path):
    ui = tmp_path / "ui.json"
    ui.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    out = tmp_path / "o.png"
    try:
        cli.main(["run", "--workflow", str(ui), "--out", str(out)])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "API format" in str(exc)


def test_run_executes_api_workflow_with_override(monkeypatch, tmp_path):
    api = tmp_path / "api.json"
    api.write_text(
        json.dumps({"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}}}),
        encoding="utf-8",
    )
    seen = {}

    def fake_run(workflow, *, base_url, timeout_sec):
        seen["workflow"] = workflow
        return Image.new("RGBA", (64, 96), (0, 0, 0, 255)), {"backend": "comfyui", "prompt_id": "x"}

    monkeypatch.setattr(cli, "run_api_workflow", fake_run)
    out = tmp_path / "o.png"
    rc = cli.main(["run", "--workflow", str(api), "--set", "6.text=new prompt", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert seen["workflow"]["6"]["inputs"]["text"] == "new prompt"


def test_list_reports_formats(tmp_path, capsys):
    (tmp_path / "api.json").write_text(
        json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8"
    )
    (tmp_path / "ui.json").write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    rc = cli.main(["list", "--dir", str(tmp_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 2
    assert payload["api_runnable"] == ["api.json"]
