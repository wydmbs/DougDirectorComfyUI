"""
comfy_client.py — thin, generic HTTP client for the ComfyUI API.

Deliberately assumes nothing about a specific workflow's node graph. Setup tab
maps node IDs; this module just queues, polls, and fetches images.
"""

import io
import json
import time
import uuid
import urllib.request
import urllib.parse
from typing import Optional


class ComfyClientError(Exception):
    pass


class ComfyClient:
    def __init__(self, base_url: str, client_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())

    def queue_prompt(self, workflow: dict) -> str:
        """POST a workflow (API-format JSON) to /prompt. Returns prompt_id."""
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ComfyClientError(f"Could not reach ComfyUI at {self.base_url}: {e}") from e
        if "error" in body:
            raise ComfyClientError(f"ComfyUI rejected the prompt: {body['error']}")
        return body["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/history/{prompt_id}", timeout=30) as resp:
            return json.loads(resp.read())

    def wait_for_completion(self, prompt_id: str, timeout_s: int = 600, poll_s: float = 2.0) -> dict:
        """Poll /history until the prompt shows up with outputs, or timeout."""
        start = time.time()
        while time.time() - start < timeout_s:
            hist = self.get_history(prompt_id)
            if prompt_id in hist:
                entry = hist[prompt_id]
                if entry.get("outputs"):
                    return entry
            time.sleep(poll_s)
        raise ComfyClientError(f"Timed out waiting for prompt {prompt_id} after {timeout_s}s")

    def fetch_image_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"{self.base_url}/view?{params}", timeout=30) as resp:
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
