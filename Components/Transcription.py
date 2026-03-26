from faster_whisper import WhisperModel
import torch
import os
import numpy as np

# Keep model reference at module level to prevent premature GC/segfault
_model_ref = None


def _normalize_audio(audio_path):
    """Normalize audio loudness for better transcription. Returns path to normalized file."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    # Target -20 dBFS — loud enough for Whisper to catch quiet audience reactions
    target_dbfs = -20.0
    change = target_dbfs - audio.dBFS
    normalized = audio.apply_gain(change)
    norm_path = os.path.splitext(audio_path)[0] + "_normalized.wav"
    normalized.export(norm_path, format="wav")
    print(f"  Audio normalized ({audio.dBFS:.1f} dBFS → {target_dbfs:.1f} dBFS)")
    return norm_path


def _detect_audience_reactions(audio_path, speech_segments):
    """Detect laughter/applause in gaps between speech segments using RMS energy.
    
    Returns a list of [marker_text, start, end] entries to merge into the transcription.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    sample_rate = audio.frame_rate

    # Convert to mono numpy array for analysis
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    # Normalize to -1..1
    peak = np.max(np.abs(samples)) or 1.0
    samples = samples / peak

    def rms_of_range(start_sec, end_sec):
        s = int(start_sec * sample_rate)
        e = int(end_sec * sample_rate)
        s = max(0, min(s, len(samples)))
        e = max(s, min(e, len(samples)))
        if e - s < 100:
            return 0.0
        chunk = samples[s:e]
        return float(np.sqrt(np.mean(chunk ** 2)))

    # Compute median speech RMS as baseline
    speech_rms_values = []
    for text, start, end in speech_segments:
        if end - start > 0.2:
            speech_rms_values.append(rms_of_range(start, end))
    if not speech_rms_values:
        return []
    median_speech_rms = float(np.median(speech_rms_values))
    # Threshold: audience reaction if gap energy > 15% of median speech energy
    reaction_threshold = median_speech_rms * 0.15

    # Find gaps between speech segments
    reactions = []
    sorted_segs = sorted(speech_segments, key=lambda s: s[1])
    for i in range(len(sorted_segs) - 1):
        gap_start = sorted_segs[i][2]  # end of current segment
        gap_end = sorted_segs[i + 1][1]  # start of next segment
        gap_duration = gap_end - gap_start

        if gap_duration < 0.3:
            continue  # gap too short

        gap_rms = rms_of_range(gap_start, gap_end)

        if gap_rms > reaction_threshold:
            # Rate loudness relative to speech
            ratio = gap_rms / median_speech_rms
            if ratio > 0.8:
                loudness = "very loud"
            elif ratio > 0.5:
                loudness = "loud"
            elif ratio > 0.3:
                loudness = "medium"
            else:
                loudness = "quiet"
            marker = f"[AUDIENCE REACTION — {loudness}, {gap_duration:.1f}s]"
            reactions.append([marker, round(gap_start, 2), round(gap_end, 2)])

    return reactions


def transcribeAudio(audio_path):
    global _model_ref
    try:
        print("Transcribing audio...")

        # Normalize audio for better detection of quiet audience reactions
        print("  Normalizing audio levels...")
        norm_path = _normalize_audio(audio_path)

        Device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {Device}")
        model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
        print("  Model loaded")
        segments, info = model.transcribe(
            audio=norm_path,
            beam_size=5,
            max_new_tokens=128,
            condition_on_previous_text=True,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,   # shorter silence counts as gap
                speech_pad_ms=200,             # pad detected speech regions
            ),
        )
        segments = list(segments)

        # Split long segments using word-level timestamps at pause boundaries.
        # Whisper with VAD can produce very long segments that merge multiple
        # sentences, hiding inter-sentence gaps where audience reactions occur.
        MIN_PAUSE_FOR_SPLIT = 0.35  # seconds of word-gap to trigger a split
        speech_segments = []
        for segment in segments:
            words = segment.words if hasattr(segment, 'words') and segment.words else None
            if not words or len(words) < 2:
                speech_segments.append([segment.text, segment.start, segment.end])
                continue
            # Walk through words and split at pauses
            sub_start = words[0].start
            sub_words = [words[0].word]
            for i in range(1, len(words)):
                gap = words[i].start - words[i - 1].end
                if gap >= MIN_PAUSE_FOR_SPLIT:
                    # Emit previous sub-segment
                    text = ''.join(sub_words).strip()
                    if text:
                        speech_segments.append([text, sub_start, words[i - 1].end])
                    sub_start = words[i].start
                    sub_words = [words[i].word]
                else:
                    sub_words.append(words[i].word)
            # Emit final sub-segment
            text = ''.join(sub_words).strip()
            if text:
                speech_segments.append([text, sub_start, words[-1].end])

        print(f"  Speech segments: {len(speech_segments)} (split from {len(segments)} Whisper segments)")

        # Detect audience reactions (laughter, applause) in gaps
        print("  Analyzing audio for audience reactions...")
        reactions = _detect_audience_reactions(audio_path, speech_segments)
        print(f"  Detected {len(reactions)} audience reactions")

        # Merge speech + reactions and sort by time
        all_segments = speech_segments + reactions
        all_segments.sort(key=lambda s: s[1])

        print(f"✓ Transcription complete: {len(speech_segments)} speech + {len(reactions)} reaction segments")

        # Clean up normalized audio
        try:
            os.remove(norm_path)
        except OSError:
            pass

        # Keep model alive at module level — ctranslate2 segfaults on CUDA cleanup
        _model_ref = model
        return all_segments
    except Exception as e:
        print("Transcription Error:", e)
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    audio_path = "audio.wav"
    transcriptions = transcribeAudio(audio_path)
    TransText = ""

    for text, start, end in transcriptions:
        TransText += (f"{start} - {end}: {text}\n")
    print(TransText)