#!/usr/bin/env python3
"""Test script for dual-speaker cropping with both layout modes."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import Components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Components.SpeakerDetection import detect_speakers_in_video
from Components.DualSpeakerCrop import crop_dual_speaker
from Components.FaceCrop import crop_to_vertical


def test_video(video_path, output_dir="test_output", test_both_layouts=True):
    """Test a single video with speaker detection and cropping."""
    if not os.path.isfile(video_path):
        print(f"Error: Video not found: {video_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem

    print(f"\n{'='*70}")
    print(f"Testing: {video_name}")
    print(f"{'='*70}")

    # Step 1: Detect speakers
    print("\n[1/3] Detecting speakers...")
    detection = detect_speakers_in_video(video_path)
    print(f"  Speakers detected: {detection['speaker_count']}")
    print(f"  Is multi-speaker: {detection['is_multi_speaker']}")
    print(f"  Confidence: {detection['confidence']:.1%}")

    # Step 2: Route based on detection
    if detection['is_multi_speaker']:
        print("\n[2/3] Processing with DUAL-SPEAKER layout...")

        layouts = ["side-by-side"]
        if test_both_layouts:
            layouts.append("split-screen")

        for layout_mode in layouts:
            output_path = os.path.join(
                output_dir,
                f"{video_name}_{layout_mode.replace('-', '_')}.mp4"
            )
            print(f"\n  Testing layout: {layout_mode}")
            print(f"  Output: {output_path}")
            try:
                crop_dual_speaker(
                    video_path,
                    output_path,
                    layout_mode=layout_mode,
                    target_height=1920,
                )
                print(f"  ✓ Success!")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                return False

    else:
        print("\n[2/3] Processing with SINGLE-SPEAKER layout...")
        output_path = os.path.join(output_dir, f"{video_name}_single_speaker.mp4")
        print(f"  Output: {output_path}")
        try:
            crop_to_vertical(
                video_path,
                output_path,
                enable_camera_effects=True,
                base_zoom=1.0,
            )
            print(f"  ✓ Success!")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    print(f"\n{'='*70}")
    print(f"✓ All tests passed for {video_name}")
    print(f"{'='*70}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test dual-speaker cropping with speaker detection"
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Video file(s) to test",
    )
    parser.add_argument(
        "-o", "--output",
        default="test_output",
        help="Output directory (default: test_output)",
    )
    parser.add_argument(
        "--single-layout",
        action="store_true",
        help="Only test side-by-side layout (skip split-screen)",
    )

    args = parser.parse_args()

    success_count = 0
    for video_path in args.videos:
        try:
            if test_video(
                video_path,
                output_dir=args.output,
                test_both_layouts=not args.single_layout,
            ):
                success_count += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nUnexpected error processing {video_path}: {e}")

    print(f"\n\nSummary: {success_count}/{len(args.videos)} videos processed successfully")
    print(f"Outputs in: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
