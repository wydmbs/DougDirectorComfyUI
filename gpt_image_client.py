"""
gpt_image_client.py -- calling GPT Image 2.5 Sunburst directly, as a cloud
alternative to Kontext for holding a character's identity.

Sunburst does the same job Kontext does -- edit an existing image against an
instruction rather than draw one from a description -- but through a
completely different mechanism: OpenAI's hosted /v1/images/edits endpoint,
not a node in the ComfyUI graph. So unlike Kontext/IP-Adapter it never
touches the workflow JSON; it replaces the render for that variant outright.
director_engine.generate() calls edit_image() directly instead of queuing a
ComfyUI prompt when reference_conditioning is set to "gpt_sunburst".

Trade-offs worth knowing before reaching for this over Kontext:
  * no seed -- the edits endpoint takes no seed parameter, so a rerun is
    never bit-identical, only similar in the way any ChatGPT redraw is.
  * cloud cost per call (OpenAI's Sunburst pricing), not local/free.
  * in exchange: a model tuned specifically for edit-instruction precision,
    worth trying on shots where Kontext's "barely rotates" limitation (see
    reference_conditioning.py) is the actual blocker.

Written by hand with urllib rather than the OpenAI SDK, matching how
harry_agent/providers/live.py talks to every other vendor in this project --
one dependency-free pattern for JSON-and-multipart HTTP, not two.
"""

import base64
import json
import os
import urllib.error
import urllib.request
import uuid

DEFAULT_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_QUALITY = "high"
DEFAULT_SIZE = "1024x1024"
EDITS_URL = "https://api.openai.com/v1/images/edits"
TIMEOUT_S = 180


class GptImageError(Exception):
    """A Sunburst edit could not be produced, with a reason worth showing."""


def _api_key() -> str:
    # Shared with harry_agent's OpenAI provider deliberately -- one key,
    # one place it's read from, rather than a second env var to keep in sync.
    return os.environ.get("OPENAI_API_KEY", "")


def _model() -> str:
    return os.environ.get("HARRY_GPT_IMAGE_MODEL", DEFAULT_MODEL)


def _quality() -> str:
    return os.environ.get("HARRY_GPT_IMAGE_QUALITY", DEFAULT_QUALITY)


def _guess_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")


def _multipart(fields: dict, file_field: str, filename: str, data: bytes, content_type: str) -> tuple:
    """Build a multipart/form-data body by hand -- one call doesn't earn a new dependency."""
    boundary = uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            .encode("utf-8")
        )
    chunks.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
         f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n').encode("utf-8")
        + data + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def edit_image(reference_image_path: str, instruction: str, model: str = "",
               quality: str = "", size: str = DEFAULT_SIZE) -> bytes:
    """Send a reference image + an editing instruction, get PNG bytes back.

    `instruction` should read the way a Kontext prompt does -- an instruction
    aimed at the picture ("turn him to face left"), not a fresh description
    of the scene -- since acting on a reference rather than redrawing it is
    the whole point of using this over a plain text-to-image call.
    """
    api_key = _api_key()
    if not api_key:
        raise GptImageError(
            "OPENAI_API_KEY is not available to this app process -- set it in the "
            "environment Harry already reads it from and restart the app.")
    if not reference_image_path or not os.path.exists(reference_image_path):
        raise GptImageError(f"Reference image not found: {reference_image_path}")
    if not (instruction or "").strip():
        raise GptImageError("No instruction was given for Sunburst to act on.")

    with open(reference_image_path, "rb") as handle:
        image_bytes = handle.read()

    fields = {
        "model": model or _model(),
        "prompt": instruction,
        "quality": quality or _quality(),
        "size": size,
        "n": "1",
    }
    body, content_type_header = _multipart(
        fields, "image", os.path.basename(reference_image_path), image_bytes,
        _guess_content_type(reference_image_path))

    request = urllib.request.Request(
        EDITS_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": content_type_header},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:800]
        raise GptImageError(f"HTTP {error.code} from Sunburst: {detail}") from error
    except urllib.error.URLError as error:
        raise GptImageError(f"Could not reach the OpenAI API: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise GptImageError("Sunburst returned something that wasn't JSON.") from error

    try:
        b64 = payload["data"][0]["b64_json"]
    except (KeyError, IndexError) as error:
        raise GptImageError(
            f"Sunburst returned no image: {json.dumps(payload)[:400]}") from error

    try:
        return base64.b64decode(b64)
    except (ValueError, TypeError) as error:
        raise GptImageError("Sunburst's image data could not be decoded.") from error


def readiness() -> dict:
    """Is there enough here to actually call Sunburst, before a run finds out the hard way."""
    return {
        "api_key_present": bool(_api_key()),
        "model": _model(),
        "quality": _quality(),
    }
