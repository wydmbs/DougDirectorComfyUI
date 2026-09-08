"""
comfy_client.py — thin, generic HTTP client for the ComfyUI API.

Deliberately assumes nothing about a specific workflow's node graph. Setup maps
node IDs; this module just uploads, queues, polls, and fetches.

On remote ComfyUI: this app may run on a laptop while ComfyUI runs on the GPU
machine. A local file path is meaningless to ComfyUI in that case, so an image
has to be *uploaded* first and the workflow then references it by the name
ComfyUI hands back. upload_image() is what makes reference conditioning work
across machines instead of silently generating with no reference at all.
"""

import json
import mimetypes
import os
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional


class ComfyClientError(Exception):
    pass


def _encode_multipart(fields: dict, file_field: str, filename: str, data: bytes,
                      content_type: str = "image/png"):
    """Build a multipart/form-data body without pulling in a dependency."""
    boundary = f"----DirectorHarness{uuid.uuid4().hex}"
    lines = []
    for key, value in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{key}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode())
    lines.append(f"--{boundary}".encode())
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode())
    lines.append(f"Content-Type: {content_type}".encode())
    lines.append(b"")
    lines.append(data)
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


class ComfyClient:
    def __init__(self, base_url: str, client_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self._upload_cache = {}

    # ----------------------------------------------------------- reachability

    def ping(self, timeout: int = 5) -> bool:
        """Is ComfyUI actually there? Cheap enough to call before a long run."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/system_stats", timeout=timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def system_stats(self, timeout: int = 10) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}/system_stats", timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise ComfyClientError(f"Could not read system stats from {self.base_url}: {e}") from e

    def object_info(self, timeout: int = 30) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}/object_info", timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise ComfyClientError(f"Could not read object_info from {self.base_url}: {e}") from e

    # ---------------------------------------------------------------- uploads

    def upload_image(self, local_path: str, subfolder: str = "director_harness",
                     overwrite: bool = True) -> str:
        """Send a local image to ComfyUI's input folder; return the name to
        reference in a LoadImage node.

        This is what lets the app drive a ComfyUI on another machine. Uploads are
        cached per (path, mtime) so reusing one turnaround sheet across eighteen
        panels doesn't re-send it eighteen times.
        """
        if not os.path.exists(local_path):
            raise ComfyClientError(f"No file to upload at {local_path}")

        try:
            stamp = os.path.getmtime(local_path)
        except OSError:
            stamp = 0
        cache_key = (os.path.abspath(local_path), stamp, subfolder)
        if cache_key in self._upload_cache:
            return self._upload_cache[cache_key]

        with open(local_path, "rb") as handle:
            data = handle.read()
        filename = os.path.basename(local_path)
        content_type = mimetypes.guess_type(filename)[0] or "image/png"

        body, header = _encode_multipart(
            {"type": "input", "subfolder": subfolder,
             "overwrite": "true" if overwrite else "false"},
            "image", filename, data, content_type)

        request = urllib.request.Request(
            f"{self.base_url}/upload/image", data=body,
            headers={"Content-Type": header}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise ComfyClientError(f"ComfyUI rejected the upload ({e.code}): {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ComfyClientError(f"Could not upload to {self.base_url}: {e}") from e

        name = result.get("name") or filename
        returned_sub = result.get("subfolder", "")
        reference = f"{returned_sub}/{name}" if returned_sub else name
        self._upload_cache[cache_key] = reference
        return reference

    # ------------------------------------------------------------------ queue

    def queue_prompt(self, workflow: dict) -> str:
        """POST a workflow (API-format JSON) to /prompt. Returns prompt_id."""
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:600]
            raise ComfyClientError(f"ComfyUI rejected the prompt ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise ComfyClientError(f"Could not reach ComfyUI at {self.base_url}: {e}") from e
        if "error" in body:
            raise ComfyClientError(f"ComfyUI rejected the prompt: {body['error']}")
        return body["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/history/{prompt_id}", timeout=30) as resp:
            return json.loads(resp.read())

    def queue_state(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}/queue", timeout=15) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return {}

    def interrupt(self) -> None:
        """Stop whatever is currently rendering."""
        try:
            request = urllib.request.Request(f"{self.base_url}/interrupt", data=b"", method="POST")
            urllib.request.urlopen(request, timeout=15)
        except (urllib.error.URLError, OSError):
            pass

    def wait_for_completion(self, prompt_id: str, timeout_s: int = 600, poll_s: float = 2.0,
                            on_progress=None) -> dict:
        """Poll /history until the prompt shows up with outputs, or timeout."""
        start = time.time()
        while time.time() - start < timeout_s:
            hist = self.get_history(prompt_id)
            if prompt_id in hist:
                entry = hist[prompt_id]
                if entry.get("outputs"):
                    return entry
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyClientError(
                        "ComfyUI reported an error running the workflow: "
                        f"{json.dumps(status)[:500]}")
            if on_progress:
                on_progress(int(time.time() - start))
            time.sleep(poll_s)
        raise ComfyClientError(f"Timed out waiting for prompt {prompt_id} after {timeout_s}s")

    # ---------------------------------------------------------------- outputs

    def fetch_image_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"{self.base_url}/view?{params}", timeout=300) as resp:
            return resp.read()

    def extract_image_refs(self, history_entry: dict) -> list:
        """Pull (filename, subfolder, folder_type) tuples for every SaveImage-style
        output node in a completed history entry."""
        refs = []
        outputs = history_entry.get("outputs", {})
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                refs.append((img["filename"], img.get("subfolder", ""), img.get("type", "output")))
        return refs

    def extract_video_refs(self, history_entry: dict) -> list:
        """Same, for video-shaped outputs. Different node packs use different
        keys, so check the ones in common use."""
        refs = []
        for node_output in (history_entry.get("outputs") or {}).values():
            for key in ("videos", "gifs", "video", "files"):
                for item in node_output.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        refs.append((item["filename"], item.get("subfolder", ""),
                                     item.get("type", "output")))
        return refs


def apply_node_overrides(workflow: dict, node_mapping, positive: str, negative: str, seed: int) -> dict:
    """Return a deep-ish copy of the workflow JSON with prompt/negative/seed
    fields overridden per the Setup tab's node mapping. Mutates a copy, not
    the original dict, so the same base workflow can be reused across calls."""
    wf = json.loads(json.dumps(workflow))  # cheap deep copy

    def set_input(node_id, input_key, value):
        if not node_id:
            return
        node = wf.get(node_id)
        if node is None:
            raise ComfyClientError(f"Node id '{node_id}' from the node mapping was not found in the workflow JSON.")
        node.setdefault("inputs", {})[input_key] = value

    set_input(node_mapping.positive_prompt_node, node_mapping.positive_prompt_input, positive)
    if node_mapping.negative_prompt_node:
        set_input(node_mapping.negative_prompt_node, node_mapping.negative_prompt_input, negative)
    if node_mapping.seed_node:
        set_input(node_mapping.seed_node, node_mapping.seed_input, seed)
    return wf
