# Commands

All commands run from the repo root in PowerShell.

## Process a local video (interactive)

```powershell
.venv\Scripts\python.exe .\main.py "E:\Videos\Predigtausschnitte\Jeden Sonntag in die Kirche - das ist doch nicht schön!.mp4"
```

## Process a local video (auto-approve, extracts ALL highlights by default)

```powershell
.venv\Scripts\python.exe .\main.py "E:\Videos\Predigtausschnitte\Jeden Sonntag in die Kirche - das ist doch nicht schön!.mp4" --auto-approve
```

Multi-highlight is now the default. This analyzes the full transcription and creates a separate short for EVERY good segment — e.g. 30 shorts from a 1-hour video, or 2-3 from a 5-minute video.

## Process a local video (single best highlight only)

```powershell
.venv\Scripts\python.exe .\main.py "E:\Videos\Predigtausschnitte\video.mp4" --single --auto-approve
```

## Process a YouTube video

```powershell
.venv\Scripts\python.exe .\main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Process every video in a YouTube playlist

```powershell
.venv\Scripts\python.exe .\playlist_runner.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" --auto-approve
```

This runs `main.py` for each video URL in the playlist, keeps resume state, and writes a per-playlist log.

## Preview playlist items without processing

```powershell
.venv\Scripts\python.exe .\playlist_runner.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" --dry-run
```

## Process a text file with YouTube URLs

Create a UTF-8 text file with one URL per line (comments starting with `#` are ignored). Example:

```text
https://www.youtube.com/watch?v=8I_PqX2LsDc
https://www.youtube.com/watch?v=3hR64GCyOg0
# comment lines are ignored
```

Run the runner to process each URL sequentially (use `--auto-approve` to skip confirmations):

```powershell
.venv\Scripts\python.exe .\urls_runner.py --file urls.txt --auto-approve
```

Use `--dry-run` to just print the parsed URLs without running `main.py`.

## Export all links from a YouTube channel to a text file

This scans the public `videos`, `streams`, and `shorts` tabs, deduplicates the watch URLs, and writes one URL per line to a UTF-8 `.txt` file.

```powershell
.venv\Scripts\python.exe .\channel_urls.py "https://www.youtube.com/@olaflatzel/videos" --output .\latzel_urls.txt
```

If you only want the channel's standard uploads tab, restrict it explicitly:

```powershell
.venv\Scripts\python.exe .\channel_urls.py "https://www.youtube.com/@olaflatzel/videos" --tabs videos --output .\latzel_videos_only.txt
```

Use `--dry-run` to print the discovered URLs without writing the file.


## Discover online videos without processing them

```powershell
.venv\Scripts\python.exe .\overnight_runner.py --preset mixed --target-count 20 --dry-run
```

This prints candidate YouTube URLs, titles, authors, and durations without downloading or generating shorts.

## Run overnight on discovered sermon + funny content

```powershell
.venv\Scripts\python.exe .\overnight_runner.py --preset mixed --target-count 200
```

This searches YouTube, skips already-completed videos using a state file, and runs `main.py --auto-approve` on each candidate until 200 successful shorts are created or the discovered queue is exhausted.

## Run overnight with your own search queries

Create a UTF-8 text file with one query per line, for example:

```text
lustige predigt deutsch
predigt humor deutsch
stand up comedy deutsch live
kabarett deutsch live
```

Then run:

```powershell
.venv\Scripts\python.exe .\overnight_runner.py --query-file .\queries.txt --target-count 200
```

## Resume an interrupted overnight run

```powershell
.venv\Scripts\python.exe .\overnight_runner.py --preset mixed --target-count 200 --state-file .\automation\overnight_state.json
```

The runner uses the state file to avoid reprocessing videos that already succeeded. Failed videos are skipped unless `--retry-failures` is added.

## Faster search, stricter source length

```powershell
.venv\Scripts\python.exe .\overnight_runner.py --preset sermons --target-count 150 --min-duration 900 --max-duration 7200 --sorts view_count upload_date
```

This is useful when you want longer long-form inputs and fewer short clips or Shorts-only results.

## Batch process multiple local files (extracts ALL highlights from each)

```powershell
Get-ChildItem "E:\Videos\Predigtausschnitte\*.mp4" | ForEach-Object { .venv\Scripts\python.exe .\main.py $_.FullName --auto-approve }
```

## Batch process, single best highlight per video

```powershell
Get-ChildItem "E:\Videos\Predigtausschnitte\*.mp4" | ForEach-Object { .venv\Scripts\python.exe .\main.py $_.FullName --single --auto-approve }
```

## Re-run (uses cached transcription automatically)

Same command as above. If a `.transcription.json` file exists next to the video, Whisper is skipped entirely.

## Delete transcription cache (force re-transcribe)

```powershell
Remove-Item "E:\Videos\Predigtausschnitte\*.transcription.json"
```

## Output location

Finished shorts are saved to `E:\Videos\Short_outputs\`.

## Overnight logs and state

- Log file: `.\automation\overnight_run.log`
- Resume state: `.\automation\overnight_state.json`
- Deferred OpenCV CUDA plan: `.\plans\opencv_cuda_plan.md`

## Environment setup

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe --pre --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision torchaudio
```

## Validate CUDA

```powershell
.venv\Scripts\python.exe -c "import torch, torch.backends.cudnn as cudnn; print('torch', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_count:', torch.cuda.device_count()); print('cudnn:', cudnn.is_available()); torch.cuda.is_available() and print('device_name:', torch.cuda.get_device_name(0))"
```
