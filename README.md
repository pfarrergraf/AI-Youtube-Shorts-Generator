# AI YouTube Shorts Generator

AI-powered tool that extracts highlight clips from long-form videos, crops them to vertical 9:16 format with DNN face tracking, adds burned-in ASS subtitles, and exports MP4 shorts. Uses a local LLM via vLLM for highlight selection and faster-whisper for GPU-accelerated transcription.

## Features

- **Multi-highlight extraction** — analyzes the full transcription and creates a separate short for every good segment (default). Use `--single` for one best clip.
- **GPU-accelerated transcription** — faster-whisper `large-v3` on CUDA with word-level timestamps and audience-reaction detection (laughter, applause).
- **Local LLM highlight selection** — local vLLM by default. LM Studio remains supported as a compatibility fallback.
- **DNN face tracking** — per-frame face detection with smooth camera effects for vertical crop.
- **ASS subtitles** — burned-in captions via ffmpeg (max 4 words/line, 2 lines). Font, size, and color are configurable.
- **NVENC encoding** — all ffmpeg steps use `h264_nvenc` for GPU-accelerated video encoding.
- **Transcription caching** — sidecar `.transcription.json` files next to source videos. Re-runs skip Whisper entirely.
- **Batch runners** — process YouTube playlists, URL lists, channel pages, or discover videos overnight via search.
- **History tracking** — persistent JSON log of downloads, created shorts, and failed attempts to avoid duplicates.

## Quick Start

### Prerequisites

- **Python 3.10** (required for faster-whisper / ctranslate2 compatibility)
- **FFmpeg** on PATH (with NVENC support for GPU encoding)
- **NVIDIA GPU** with CUDA support
- **Local vLLM server** running on `localhost:1234` (or adjust the `.env` values)
- [**uv**](https://docs.astral.sh/uv/) package manager

### Installation

```powershell
git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
cd AI-Youtube-Shorts-Generator

# Create venv and install dependencies
uv venv .venv --python 3.10
uv sync --frozen

# Install PyTorch with CUDA (try cu130 nightly first, fall back to cu128)
uv pip install --python .venv\Scripts\python.exe --pre --upgrade --force-reinstall `
    --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision torchaudio

# Validate CUDA
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Environment

Create a `.env` file in the project root:

```env
VLLM_API_KEY=local-vllm
VLLM_BASE_URL=http://127.0.0.1:1234/v1
VLLM_MODEL=qwen2.5-72b-instruct
```

Legacy compatibility:
- `OPENAI_API`, `OPENAI_BASE_URL`, and `LLM_MODEL` are still accepted
- but `VLLM_*` is now the preferred local-server configuration

## Usage

All commands assume you are in the `parakeet_uv/` directory with the venv activated
(`source .venv/bin/activate`) or using `.venv/bin/python` directly.

### Process a single video (extracts ALL highlights)

```bash
.venv/bin/python cli/parakeet.py shorts \
  --input "https://www.youtube.com/watch?v=VIDEO_ID" --auto-approve
```

### Process a local file

```bash
# On WSL, translate Windows drive letters: E:\... → /mnt/e/...
.venv/bin/python cli/parakeet.py shorts \
  --input "/mnt/e/Videos/my_video.mp4" --auto-approve --karaoke-subtitles
```

### Single best highlight only

```bash
.venv/bin/python cli/parakeet.py shorts \
  --input "/mnt/e/Videos/my_video.mp4" --single --auto-approve
```

### Process a YouTube playlist

```bash
.venv/bin/python playlist_runner.py \
  "https://www.youtube.com/playlist?list=PLAYLIST_ID" --auto-approve
```

### Process a list of URLs from a text file

```bash
.venv/bin/python urls_runner.py --file urls.txt --auto-approve
```

### Extract URLs from a YouTube channel

```bash
.venv/bin/python channel_urls.py \
  "https://www.youtube.com/@ChannelName/videos" --output channel_urls.txt
```

### Overnight autonomous batch

```bash
# Discover and process sermon + comedy videos (min 200K views)
.venv/bin/python overnight_runner.py --preset mixed --target-count 200

# Dry-run: preview candidates without processing
.venv/bin/python overnight_runner.py --preset sermons --dry-run
```

### Batch process local files

```bash
for f in /mnt/e/Videos/*.mp4; do
  .venv/bin/python cli/parakeet.py shorts --input "$f" --auto-approve
done
```

## How It Works

1. **Download** — fetches from YouTube (pytubefix) or uses a local file
2. **Extract audio** — WAV via ffmpeg
3. **Transcribe** — faster-whisper `large-v3` on CUDA with word timestamps + audience reaction detection
4. **Cache** — saves transcription as sidecar `.transcription.json` (skipped on re-run)
5. **Highlight selection** — local LLM analyzes transcription, returns all good clip boundaries (30-120s each)
6. **Smart padding** — extends clip end into trailing silence/laughter without bleeding into next segment
7. **Extract clip** — NVENC-accelerated crop of the selected timeframe
8. **Vertical crop** — DNN face tracking for 9:16 format with smooth camera motion
9. **Subtitles** — burns ASS captions via ffmpeg (NVENC)
10. **Audio merge** — combines original audio with the subtitled vertical clip
11. **Cleanup** — removes all temp files, writes history

**Output**: `{slugified-title}_{session}_{clip}_short.mp4`

## Project Structure

```text
main.py                  # Pipeline orchestrator
search_runner.py         # YouTube search -> batch process
overnight_runner.py      # Autonomous overnight discovery + processing
urls_runner.py           # Process URLs from a text file
playlist_runner.py       # Process YouTube playlists
channel_urls.py          # Extract URLs from a channel page

Components/
  Edit.py                # Audio extraction, clip cutting (NVENC)
  Transcription.py       # faster-whisper transcription + audience reactions
  LanguageTasks.py       # LLM highlight selection + local server checks
  FaceCrop.py            # 9:16 vertical crop with DNN face tracking
  Subtitles.py           # ASS subtitle burning (ffmpeg NVENC)
  YoutubeDownloader.py   # YouTube download via pytubefix
  History.py             # JSON history tracking

models/                  # DNN face detector weights
```

## Configuration

All settings are in `.env`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_API_KEY` | `local-vllm` | Local vLLM API key |
| `VLLM_BASE_URL` | `http://127.0.0.1:1234/v1` | Local vLLM endpoint |
| `VLLM_MODEL` | `qwen2.5-72b-instruct` | Model identifier |

Legacy aliases:
- `OPENAI_API` / `OPENAI_API_KEY`
- `OPENAI_BASE_URL` / `OPENAI_API_BASE`
- `LLM_MODEL`

Output directory is set at the top of `main.py` (`OUTPUT_DIR`).

## CPU-Only Installation

See [INSTALL_CPU.md](INSTALL_CPU.md) for setup without an NVIDIA GPU.

## License

MIT
