"""Explicit Azure Foundry GPT-image-2 generation jobs.

Normal thumbnail rendering must stay deterministic and repertoire-first.  This
module is deliberately a job provider: it generates a candidate only when a
caller explicitly requests it and never edits the speaker manifest.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_ROOT / ".env"


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return ""


def _normalise_base_url(endpoint: str) -> str:
    """Accept a project endpoint, v1 endpoint, or full generations URL."""
    value = endpoint.rstrip("/")
    for suffix in ("/images/generations", "/images/edits"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.rstrip("/") + "/"


@dataclass(frozen=True)
class AzureImageConfig:
    api_key: str
    base_url: str
    deployment_name: str

    @classmethod
    def from_env(cls, env_path: str | Path = _ENV_PATH) -> "AzureImageConfig":
        # Explicitly load the generator's .env, regardless of the caller's cwd.
        load_dotenv(Path(env_path), override=False)
        api_key = _first_env(
            "AZURE_OPENAI_API_KEY",
            "AZURE_API_KEY",
            "api_key",
        )
        endpoint = _first_env(
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_ENDPOINT",
            "FOUNDRY_PROJECT_ENDPOINT",
            "endpoint",
        )
        deployment_name = _first_env(
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_IMAGE_DEPLOYMENT",
            "deployment_name",
            "gpt-image-2",
        )
        missing = [
            name for name, value in (
                ("API key", api_key),
                ("endpoint", endpoint),
                ("deployment name", deployment_name),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                "Azure image generation is not configured; missing "
                + ", ".join(missing)
                + f" in {_ENV_PATH}"
            )
        return cls(
            api_key=api_key,
            base_url=_normalise_base_url(endpoint),
            deployment_name=deployment_name,
        )


def _client(config: AzureImageConfig):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Install the existing 'openai' project dependency first.") from exc
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _extract_b64(response) -> str:
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError("Azure image response contained no image data")
    encoded = getattr(data[0], "b64_json", None)
    if not encoded:
        raise RuntimeError("Azure image response contained no b64_json payload")
    return encoded


def generate_image(
    prompt: str,
    output_path: str | Path,
    *,
    reference_images: Iterable[str | Path] = (),
    size: str = "1024x1536",
    quality: str = "high",
    config: AzureImageConfig | None = None,
) -> Path:
    """Generate one text-free candidate, optionally using reference images.

    The output path is written exactly as requested.  No manifest or approved
    flag is changed; candidates remain reviewable until promoted manually.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    config = config or AzureImageConfig.from_env()
    client = _client(config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    refs = [Path(path) for path in reference_images]

    if refs:
        handles = []
        try:
            for path in refs:
                if not path.is_file():
                    raise FileNotFoundError(path)
                handles.append((path.name, path.read_bytes(), "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"))
            response = client.images.edit(
                image=handles,
                prompt=prompt,
                model=config.deployment_name,
                n=1,
                size=size,
                quality=quality,
            )
        finally:
            # The SDK consumes bytes here; no open file descriptors to close.
            handles.clear()
    else:
        response = client.images.generate(
            prompt=prompt,
            model=config.deployment_name,
            n=1,
            size=size,
            quality=quality,
            output_format="png",
        )

    output.write_bytes(base64.b64decode(_extract_b64(response)))
    return output


def speaker_source_paths(speaker_key: str) -> list[Path]:
    """Return local identity source images listed for a manifest speaker."""
    manifest_path = _PROJECT_ROOT / "assets" / "speaker_references" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("speakers", {}).get(speaker_key, {}).get("source_images", [])
    root = manifest_path.parent
    return [root / str(item["path"]) for item in entries if isinstance(item, dict) and item.get("path")]

