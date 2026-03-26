import sys
import os
import uuid
import re
import json

# Configure ImageMagick 7 path before any moviepy import
_im_path = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
if os.path.exists(_im_path):
    os.environ["IMAGEMAGICK_BINARY"] = _im_path

from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight, GetAllHighlights
from Components.FaceCrop import crop_to_vertical, combine_videos
from Components.Subtitles import add_subtitles_to_video
from Components.History import is_downloaded, is_short_created, mark_downloaded, mark_short_created, mark_failed

# Output directory for finished shorts
OUTPUT_DIR = r"E:\Videos\Short_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _smart_pad_end(clip_end, transcriptions, max_pad=3.0):
    """Pad the clip end to include trailing silence/laughter but STOP before new
    speech content starts.  Returns the padded end time.

    Strategy:
    - Find the first speech segment that starts AFTER clip_end.
    - Allow padding up to (that segment's start - 0.1s), but no more than max_pad.
    - This prevents bleeding into the next joke/topic.
    """
    next_speech_start = None
    for text, start, end in transcriptions:
        if start >= clip_end and not str(text).startswith("["):
            next_speech_start = start
            break

    if next_speech_start is not None:
        available = next_speech_start - clip_end - 0.1  # leave 0.1s gap
        pad = max(0.0, min(available, max_pad))
    else:
        pad = max_pad  # no next speech segment, safe to pad fully

    return clip_end + pad

# Generate unique session ID for this run (for concurrent execution support)
session_id = str(uuid.uuid4())[:8]
print(f"Session ID: {session_id}")

# Check for auto-approve flag (for batch processing)
auto_approve = "--auto-approve" in sys.argv
if auto_approve:
    sys.argv.remove("--auto-approve")

# Check for --all flag (extract ALL highlights — this is the DEFAULT now)
# Use --single to extract only the single best highlight instead
extract_all_flag = "--single" not in sys.argv
if "--all" in sys.argv:
    sys.argv.remove("--all")
if "--single" in sys.argv:
    sys.argv.remove("--single")

# Check if URL/file was provided as command-line argument
if len(sys.argv) > 1:
    url_or_file = sys.argv[1]
    print(f"Using input from command line: {url_or_file}")
else:
    url_or_file = input("Enter YouTube video URL or local video file path: ")

# Check if input is a local file
video_title = None
if os.path.isfile(url_or_file):
    print(f"Using local video file: {url_or_file}")
    Vid = url_or_file
    # Extract title from filename
    video_title = os.path.splitext(os.path.basename(url_or_file))[0]
    mark_downloaded(url_or_file, title=video_title, local_path=url_or_file)
else:
    # Check if already downloaded
    if is_short_created(url_or_file):
        print(f"Short already created for this video. Skipping.")
        os._exit(0)
    # Assume it's a YouTube URL
    print(f"Downloading from YouTube: {url_or_file}")
    Vid = download_youtube_video(url_or_file)
    if Vid:
        Vid = Vid.replace(".webm", ".mp4")
        print(f"Downloaded video and audio files successfully! at {Vid}")
        # Extract title from downloaded file path
        video_title = os.path.splitext(os.path.basename(Vid))[0]
        mark_downloaded(url_or_file, title=video_title, local_path=Vid)

# Clean and slugify title for filename
def clean_filename(title):
    # Convert to lowercase
    cleaned = title.lower()
    # Remove or replace invalid filename characters
    cleaned = re.sub(r'[<>:"/\\|?*\[\]]', '', cleaned)
    # Replace spaces and underscores with hyphens
    cleaned = re.sub(r'[\s_]+', '-', cleaned)
    # Remove multiple consecutive hyphens
    cleaned = re.sub(r'-+', '-', cleaned)
    # Remove leading/trailing hyphens
    cleaned = cleaned.strip('-')
    # Limit length
    return cleaned[:80]

# Process video (works for both local files and downloaded videos)
if Vid:
    # Create temporary audio filename
    audio_file = f"audio_{session_id}.wav"
    
    try:
        # Check for cached transcription next to the source video
        cache_file = os.path.splitext(Vid)[0] + ".transcription.json"
        
        if os.path.exists(cache_file):
            print(f"Loading cached transcription from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                transcriptions = json.load(f)
            print(f"✓ Loaded {len(transcriptions)} cached segments")
        else:
            Audio = extractAudio(Vid, audio_file)
            if not Audio:
                print("No audio file found")
                sys.exit(1)

            transcriptions = transcribeAudio(Audio)
            if len(transcriptions) == 0:
                print("No transcriptions found")
                sys.exit(1)
            
            # Save transcription cache
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(transcriptions, f, ensure_ascii=False, indent=2)
            print(f"✓ Transcription cached to {cache_file}")

        print(f"\n{'='*60}")
        print(f"TRANSCRIPTION SUMMARY: {len(transcriptions)} segments")
        print(f"{'='*60}\n")
        TransText = ""

        for text, start, end in transcriptions:
            TransText += (f"{start} - {end}: {text}\n")

        print("Analyzing transcription to find best highlight...")

        # Check for --all flag to extract ALL highlights from the video
        extract_all = extract_all_flag

        if extract_all:
            # Multi-highlight mode: find ALL good clips in the video
            highlights = GetAllHighlights(TransText)
            if not highlights:
                print("No highlights found in this video.")
                mark_failed(url_or_file, error="No highlights found")
                sys.exit(1)

            if not auto_approve:
                print(f"\nFound {len(highlights)} highlight(s). Process all?")
                user_input = input("[Enter/y] Process all  [n] Cancel: ").strip().lower()
                if user_input == 'n':
                    print("Cancelled by user")
                    sys.exit(0)
        else:
            # Single-highlight mode (original behavior)
            start, stop = GetHighlight(TransText)
            if start is None or stop is None:
                print("ERROR: Failed to get highlight from LLM")
                mark_failed(url_or_file, error="Highlight extraction failed")
                sys.exit(1)

            approved = auto_approve
            if not auto_approve:
                while not approved:
                    print(f"\n{'='*60}")
                    print(f"SELECTED SEGMENT: {start}s - {stop}s ({stop-start}s duration)")
                    print(f"{'='*60}\n")
                    user_input = input("[Enter/y] Approve  [r] Regenerate  [n] Cancel: ").strip().lower()
                    if user_input == 'r':
                        start, stop = GetHighlight(TransText)
                        if start is None or stop is None:
                            print("Failed to regenerate. Exiting.")
                            sys.exit(1)
                    elif user_input == 'n':
                        print("Cancelled by user")
                        sys.exit(0)
                    else:
                        approved = True
            else:
                print(f"SELECTED SEGMENT: {start}s - {stop}s ({stop-start}s duration) — auto-approved")

            highlights = [{"start": start, "end": stop, "content": "", "quality": "high"}]

        # Process each highlight
        total_shorts = 0
        for clip_idx, highlight in enumerate(highlights):
            clip_start = highlight["start"]
            clip_end = highlight["end"]

            if clip_start < 0 or clip_end <= 0 or clip_end <= clip_start:
                print(f"  Skipping clip {clip_idx+1}: invalid times ({clip_start} - {clip_end})")
                continue

            clip_session = f"{session_id}_{clip_idx}"
            temp_clip = f"temp_clip_{clip_session}.mp4"
            temp_cropped = f"temp_cropped_{clip_session}.mp4"
            temp_subtitled = f"temp_subtitled_{clip_session}.mp4"
            clip_temps = [temp_clip, temp_cropped, temp_subtitled]

            try:
                pad_before = 0.5
                padded_start = max(0, clip_start - pad_before)
                padded_stop = _smart_pad_end(clip_end, transcriptions, max_pad=3.0)

                prefix = f"[{clip_idx+1}/{len(highlights)}] " if len(highlights) > 1 else ""
                print(f"\n{prefix}Creating short: {padded_start:.1f}s - {padded_stop:.1f}s ({padded_stop-padded_start:.1f}s)")

                print(f"{prefix}Step 1/4: Extracting clip...")
                crop_video(Vid, temp_clip, padded_start, padded_stop)

                print(f"{prefix}Step 2/4: Cropping to vertical (9:16)...")
                crop_to_vertical(temp_clip, temp_cropped)

                print(f"{prefix}Step 3/4: Adding subtitles...")
                add_subtitles_to_video(temp_cropped, temp_subtitled, transcriptions, video_start_time=padded_start)

                clean_title = clean_filename(video_title) if video_title else "output"
                if len(highlights) > 1:
                    final_output = os.path.join(OUTPUT_DIR, f"{clean_title}_{clip_session}_short.mp4")
                else:
                    final_output = os.path.join(OUTPUT_DIR, f"{clean_title}_{session_id}_short.mp4")

                print(f"{prefix}Step 4/4: Muxing audio...")
                combine_videos(temp_clip, temp_subtitled, final_output)

                abs_path = os.path.abspath(final_output)
                file_uri = 'file:///' + abs_path.replace('\\', '/')
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS: {final_output}")
                print(f"  {file_uri}")
                print(f"{'='*60}\n")
                mark_short_created(url_or_file, output_path=final_output)
                total_shorts += 1

            except Exception as clip_err:
                print(f"Error processing clip {clip_idx+1}: {clip_err}")
                import traceback
                traceback.print_exc()
            finally:
                for tf in clip_temps:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except OSError:
                        pass

        if total_shorts == 0:
            mark_failed(url_or_file, error="All clips failed")
        print(f"\nFinished: {total_shorts}/{len(highlights)} shorts created")

    finally:
        # Clean up audio temp file
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except OSError:
            pass
        print(f"Cleaned up temporary files for session {session_id}")

# Force-exit to prevent ctranslate2 CUDA segfault during Python GC at shutdown
os._exit(0)
