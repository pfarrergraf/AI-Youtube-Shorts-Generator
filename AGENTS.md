# Project Guidelines

## Overview

AI-powered tool that extracts highlight clips from long-form videos (primarily German sermons and comedy), crops them to vertical 9:16 format with DNN face tracking, adds ASS subtitles, and exports MP4 shorts. Uses a local LLM via vLLM for highlight selection and faster-whisper for GPU-accelerated transcription.

## Architecture

```text
main.py                  # Pipeline orchestrator: input -> audio -> transcribe -> highlight(s) -> crop -> subtitle -> export
search_runner.py         # YouTube search -> batch process top results
overnight_runner.py      # Autonomous overnight batch: discover videos via YouTube search, filter by views, process sequentially
urls_runner.py           # Process a text file of YouTube URLs sequentially
playlist_runner.py       # Process every video in a YouTube playlist
channel_urls.py          # Extract all video URLs from a YouTube channel page

Components/
  Edit.py                # Audio extraction and clip cutting (NVENC)
  Transcription.py       # Speech-to-text via faster-whisper (large-v3, CUDA) + audience reaction detection
  LanguageTasks.py       # LLM highlight selection + local server checks + retry logic
  FaceCrop.py            # 9:16 vertical crop with DNN face tracking + camera effects + audio merge
  Subtitles.py           # Burn-in ASS captions via ffmpeg (NVENC)
  YoutubeDownloader.py   # YouTube download via pytubefix + ffmpeg merge
  History.py             # Persistent JSON history of downloads + created shorts
  Speaker.py             # Legacy (not used)
  SpeakerDetection.py    # Legacy (not used)
  TextOverlay.py         # Legacy (not used)

models/                  # DNN face detector weights (deploy.prototxt + caffemodel)
automation/              # Runtime state: history JSON, overnight state, logs (gitignored)

Components/ComfyUIBackground.py   # ComfyUI bridge: backgrounds, img2img edits, generic API-workflow runner
comfyui_bridge_cli.py             # CLI over the bridge: status | background | edit | run | list
```

## ComfyUI Bridge

`Components/ComfyUIBackground.py` talks to a running ComfyUI server (default `http://127.0.0.1:8188`, served from `/home/benjamin_graf/ComfyUI`, auto-detected via `~/ComfyUI` or `PARAKEET_COMFYUI_ROOT`).

- `generate_background_image(...)` — text→image; auto-wired into the `bold_minimal` thumbnail template in `Components/ThumbnailMoveChurch.py`.
- `generate_edited_image(...)` — img2img restyle/edit; uploads the frame via `/upload/image`, runs an API-format graph on `sd_xl_turbo_1.0_fp16.safetensors`.
- `run_api_workflow(workflow_dict, ...)` — submit any **API-format** workflow and ingest the output image.
- `check_server(...)`, `detect_workflow_format(path)` (returns `api` | `ui` | `other`).

Run the CLI from this dir (`python comfyui_bridge_cli.py status`). Verify with `pytest tests/test_comfyui_edit.py tests/test_comfyui_bridge_cli.py -q` (server-free). All entry points fall back gracefully when the server is down (background → procedural gradient, edit → unmodified source).

**Constraint:** only API-format JSON (flat `{node_id: {class_type, inputs}}`) runs via `/prompt`; UI-graph exports (`blueprints/*.json`, most `C:\ComfyUI\...\workflows`) do not — re-save them via **Save (API Format)** first.

## Speaker hero repertoire

Use `assets/speaker_references/manifest.json` as the source of truth for
speaker identity references and reviewed, text-free AI hero plates. Reuse a
speaker hero before generating a new one. The image model must never render
the final headline; add exact typography afterward with Pillow. Codex's hosted
imagegen does not use the workstation GPU and is not callable from this app's
internal quota. App-triggered generation must use an explicit local ComfyUI or
separate OpenAI API provider, save the result under the speaker's repertoire,
and pass an identity/readability quality gate before promotion.

## Build and Test

Run these commands from the repository root in PowerShell.

```powershell
uv venv .venv --python 3.10
uv sync --frozen
uv pip install --python .venv\Scripts\python.exe --pre --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision torchaudio
.venv\Scripts\python.exe -c 'import torch, torch.backends.cudnn as cudnn; print("torch", torch.__version__); print("cuda_available:", torch.cuda.is_available()); print("cuda_count:", torch.cuda.device_count()); print("cudnn:", cudnn.is_available()); torch.cuda.is_available() and print("device_name:", torch.cuda.get_device_name(0))'
```

If the `cu130` install succeeds but the validation command does not report CUDA correctly, reinstall the torch stack with the validated `cu128` fallback and rerun the same validation command:

```powershell
uv pip install --python .venv\Scripts\python.exe --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
.venv\Scripts\python.exe -c 'import torch, torch.backends.cudnn as cudnn; print("torch", torch.__version__); print("cuda_available:", torch.cuda.is_available()); print("cuda_count:", torch.cuda.device_count()); print("cudnn:", cudnn.is_available()); torch.cuda.is_available() and print("device_name:", torch.cuda.get_device_name(0))'
```

After the torch stack validates on either path, run the repo smoke tests:

```powershell
.venv\Scripts\python.exe -c "from Components.LanguageTasks import GetHighlight; print('LLM module OK')"
.venv\Scripts\python.exe .\main.py 'path\to\video.mp4' --auto-approve
```

## Thumbnail V2 production rule

V2 ist der Produktionsstandard. Der Textblock muss vollständig außerhalb der
konservativen Face-Safe-Zone liegen; Varianten mit Face-Overlap werden hart
verworfen. `final_ab_v2` ist die visuelle Referenz: Sprecher/Hero auf einer
Seite, Hook klar auf der anderen. Textfreie Sprecher-Heroes liegen in
`assets/speaker_references/` und werden deterministisch aus dem Manifest gewählt.

## Conventions

- **Host OS**: Assume Windows 11 first. Prefer PowerShell commands unless WSL is specifically needed.
- **Package manager**: Always `uv`, never `pip` or `python -m pip`.
- **Virtual env**: Always `.venv`, created via `uv venv`. If `.venv` is missing `pyvenv.cfg` or otherwise broken, recreate it instead of trying to repair it in place.
- **Dependency management**: `pyproject.toml` is the single source of truth for dependencies. `requirements.txt` is kept for backward compat but `uv sync --frozen` (which reads `uv.lock`) is the preferred install path. After adding/removing deps, run `uv lock --python 3.10` to regenerate the lockfile.
- **Python version**: Use Python 3.10 for this repo. This is a repo-specific exception to the global Python 3.11 default because this project is validated around `faster-whisper` and `ctranslate2`.
- **PyTorch / CUDA**: CUDA is required. Try `cu130` first, then fall back to `cu128` only if `cu130` fails validation. Do not restore the old `cu126` guidance.
- **Torch install order**: Install project requirements, then explicitly reinstall `torch`, `torchvision`, and `torchaudio` from the chosen CUDA index so the final environment is not left on an ambiguous wheel source.
- **Torch validation**: Always use the standard one-line CUDA test command above after installing or changing torch.
- **CUDA policy**: Never switch this repo to CPU as the default path. If CUDA is broken, debug the issue instead.
- **Validated torch stacks on this machine**: Preferred first: `2.12.0.dev* + cu130`. Stable fallback: `2.9.1+cu128`.
- **LLM backend**: local vLLM at `localhost:1234` by default. Configure via `.env` vars `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL`. The old `OPENAI_*` names still work as compatibility aliases. LM Studio is only the fallback mode now.
- **Whisper model**: `large-v3` (multilingual, auto-detects language). Runs on GPU alongside the LLM.
- **Primary language**: German (Predigten / sermons). Code must not hardcode `language="en"` anywhere.
- **requirements.txt**: Keep direct dependencies only. Never commit a frozen transitive dependency dump. `pyproject.toml` is authoritative; `requirements.txt` is a convenience mirror.
- **Clip duration**: Target 30-120 seconds per clip.
- **Multi-highlight**: Default behavior extracts ALL highlights from a video. Use `--single` for legacy single-best mode.
- **Face tracking**: Dynamic per-frame DNN tracking, not static first-30-frames crop.
- **Temp files**: Use session-ID-suffixed filenames and clean them up after export.
- **Output**: Local MP4 only. No upload automation.
- **Video encoding**: NVENC (h264_nvenc) on GPU for all encoding steps.
- **Subtitles**: ASS format burned via ffmpeg. Max 4 words/line, 2 lines.

## Common Pitfalls

- `select.select()` on stdin does not work on Windows. Use `input()` for interactive prompts.
- MoviePy 1.0.3 `TextClip` requires ImageMagick installed and available on `PATH`.
- Some local servers may not support `function_calling` structured output. The code already has fallbacks to `json_mode` and raw parsing.
- OpenCV `VideoWriter` on Windows needs `XVID`, not `mp4v`.
- `Components/Speaker.py` and `Components/SpeakerDetection.py` are legacy files. Do not import them in the main pipeline.
- A torch install that imports successfully is not enough. Treat `torch.cuda.is_available()`, `torch.cuda.device_count()`, and `cudnn.is_available()` as the minimum validation gate.
