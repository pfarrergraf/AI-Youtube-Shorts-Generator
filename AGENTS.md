# Project Guidelines

## Overview

AI-powered tool that extracts highlight clips from long-form videos (primarily German sermons and comedy), crops them to vertical 9:16 format with DNN face tracking, adds ASS subtitles, and exports MP4 shorts. Uses a local LLM (via LM Studio) for highlight selection and faster-whisper for GPU-accelerated transcription.

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
  LanguageTasks.py       # LLM highlight selection + LM Studio auto-start/auto-load + retry logic
  FaceCrop.py            # 9:16 vertical crop with DNN face tracking + camera effects + audio merge
  Subtitles.py           # Burn-in ASS captions via ffmpeg (NVENC)
  YoutubeDownloader.py   # YouTube download via pytubefix + ffmpeg merge
  History.py             # Persistent JSON history of downloads + created shorts
  Speaker.py             # Legacy (not used)
  SpeakerDetection.py    # Legacy (not used)
  TextOverlay.py         # Legacy (not used)

models/                  # DNN face detector weights (deploy.prototxt + caffemodel)
automation/              # Runtime state: history JSON, overnight state, logs (gitignored)
```

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
- **LLM backend**: LM Studio at `localhost:1234`. Configurable via `.env` vars `OPENAI_BASE_URL`, `OPENAI_API`, `LLM_MODEL`. The code auto-starts LM Studio and auto-loads the configured model via the `lms` CLI if they are not running.
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
- LM Studio may not support `function_calling` structured output. The code already has fallbacks to `json_mode` and raw parsing.
- OpenCV `VideoWriter` on Windows needs `XVID`, not `mp4v`.
- `Components/Speaker.py` and `Components/SpeakerDetection.py` are legacy files. Do not import them in the main pipeline.
- A torch install that imports successfully is not enough. Treat `torch.cuda.is_available()`, `torch.cuda.device_count()`, and `cudnn.is_available()` as the minimum validation gate.
