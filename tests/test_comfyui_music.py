from Components.ComfyUIMusic import (
    _extract_output_audio,
    build_ace_step_workflow,
)


def test_build_ace_step_workflow_uses_native_turbo_recipe():
    workflow = build_ace_step_workflow(
        tags="warm instrumental piano, no vocals",
        duration_sec=30,
        bpm=76,
        seed=42,
    )

    assert workflow["model"]["inputs"]["unet_name"] == "acestep_v1.5_turbo.safetensors"
    assert workflow["positive"]["inputs"]["lyrics"] == "[Instrumental]"
    assert workflow["positive"]["inputs"]["generate_audio_codes"] is False
    assert workflow["positive"]["inputs"]["duration"] == 30
    assert workflow["sampler"]["inputs"]["steps"] == 8
    assert workflow["sampler"]["inputs"]["cfg"] == 1.0
    assert workflow["sampling"]["inputs"]["shift"] == 3.0
    assert workflow["level"]["inputs"]["volume"] == -3
    assert workflow["save"]["inputs"]["audio"] == ["level", 0]
    assert workflow["save"]["class_type"] == "SaveAudio"


def test_build_ace_step_workflow_supports_mp3():
    workflow = build_ace_step_workflow(
        tags="ambient instrumental",
        duration_sec=20,
        bpm=70,
        seed=7,
        audio_format="mp3",
    )

    assert workflow["save"]["class_type"] == "SaveAudioMP3"
    assert workflow["save"]["inputs"]["quality"] == "320k"


def test_build_ace_step_workflow_can_enable_lm_audio_codes():
    workflow = build_ace_step_workflow(
        tags="continuous ambient instrumental",
        duration_sec=30,
        bpm=72,
        seed=8,
        generate_audio_codes=True,
    )

    assert workflow["positive"]["inputs"]["generate_audio_codes"] is True


def test_extract_output_audio_from_comfy_history():
    entry = {
        "outputs": {
            "save": {
                "audio": [
                    {
                        "filename": "parakeet_music_00001_.flac",
                        "subfolder": "audio",
                        "type": "output",
                    }
                ]
            }
        }
    }

    assert _extract_output_audio(entry) == {
        "filename": "parakeet_music_00001_.flac",
        "subfolder": "audio",
        "type": "output",
    }
