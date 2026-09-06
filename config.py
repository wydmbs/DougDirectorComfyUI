"""
config.py — shared persisted configuration for the ComfyUI Director Harness.

One small JSON file, edited once via the Setup tab, reused/extended by every
module (image build, future video module, future QA module) rather than each
module inventing its own config file. See SUITE.md.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

CONFIG_PATH = os.environ.get("TOOLCHAIN_CONFIG_PATH", "toolchain_config.json")


@dataclass
class NodeMapping:
    """Which node IDs in the exported workflow JSON hold the fields we drive."""
    positive_prompt_node: str = ""
    positive_prompt_input: str = "text"
    negative_prompt_node: str = ""
    negative_prompt_input: str = "text"
    seed_node: str = ""
    seed_input: str = "seed"
    save_image_node: str = ""


@dataclass
class ToolchainConfig:
    comfyui_url: str = "http://127.0.0.1:8188"
    workflow_json_path: str = ""
    node_mapping: NodeMapping = field(default_factory=NodeMapping)
    storyboard_path: str = "storyboard.xlsx"
    images_dir: str = "images"
    mock_mode: bool = True

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ToolchainConfig":
        nm = d.get("node_mapping", {}) or {}
        cfg = ToolchainConfig(
            comfyui_url=d.get("comfyui_url", "http://127.0.0.1:8188"),
            workflow_json_path=d.get("workflow_json_path", ""),
            node_mapping=NodeMapping(**nm) if nm else NodeMapping(),
            storyboard_path=d.get("storyboard_path", "storyboard.xlsx"),
            images_dir=d.get("images_dir", "images"),
            mock_mode=d.get("mock_mode", True),
        )
        return cfg


def load_config(path: str = CONFIG_PATH) -> ToolchainConfig:
    if not os.path.exists(path):
        return ToolchainConfig()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return ToolchainConfig.from_dict(json.load(f))
    except (json.JSONDecodeError, OSError):
        return ToolchainConfig()


def save_config(cfg: ToolchainConfig, path: str = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
