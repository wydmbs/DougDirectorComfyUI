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
class AzureConfig:
    """Azure OpenAI settings owned by this app.

    Previously Harry read these out of an unrelated application's config file on
    disk, which meant Azure silently stopped working on any machine that didn't
    happen to have that other app installed. The harness owns its own settings.
    """
    endpoint: str = ""
    deployment: str = ""
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version: str = "2024-10-21"

    def is_ready(self) -> bool:
        return bool(self.endpoint and self.deployment and os.environ.get(self.api_key_env))


@dataclass
class VideoConfig:
    """Where the clip-stage models live and what they're called.

    Endpoints and model names are config rather than constants because these
    services rename and re-version far faster than this app will be rebuilt.
    """
    minimax_url: str = "https://api.minimax.chat/v1"
    minimax_model: str = "MiniMax-Hailuo-H3"
    minimax_api_key_env: str = "MINIMAX_API_KEY"

    ltx_workflow_path: str = ""

    runway_url: str = "https://api.dev.runwayml.com/v1"
    runway_model: str = "gen4_turbo"
    runway_version: str = "2024-11-06"
    runway_api_key_env: str = "RUNWAY_API_KEY"


@dataclass
class AgentConfig:
    """How much rope the assistant director gets when running on its own."""
    max_steps: int = 40
    auto_lock: bool = False
    auto_install: bool = False
    variants_per_generation: int = 4
    critique_enabled: bool = True
    max_iterations_per_asset: int = 3
    # FLUX.1 Dev + IP-Adapter is the chosen keyframe path, so identity locking
    # is on by default: without it every character quietly drifts between shots.
    reference_conditioning: str = "ipadapter"
    ipadapter_weight: float = 0.8


@dataclass
class ToolchainConfig:
    comfyui_url: str = "http://127.0.0.1:8188"
    workflow_json_path: str = ""
    node_mapping: NodeMapping = field(default_factory=NodeMapping)
    storyboard_path: str = "storyboard.xlsx"
    images_dir: str = "images"
    mock_mode: bool = True
    harry_provider: str = "Claude"
    active_project_id: str = ""
    azure: AzureConfig = field(default_factory=AzureConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ToolchainConfig":
        nm = d.get("node_mapping", {}) or {}
        az = d.get("azure", {}) or {}
        ag = d.get("agent", {}) or {}
        vd = d.get("video", {}) or {}
        cfg = ToolchainConfig(
            comfyui_url=d.get("comfyui_url", "http://127.0.0.1:8188"),
            workflow_json_path=d.get("workflow_json_path", ""),
            node_mapping=NodeMapping(**nm) if nm else NodeMapping(),
            storyboard_path=d.get("storyboard_path", "storyboard.xlsx"),
            images_dir=d.get("images_dir", "images"),
            mock_mode=d.get("mock_mode", True),
            harry_provider=d.get("harry_provider", "Claude"),
            active_project_id=d.get("active_project_id", ""),
            azure=AzureConfig(**{k: v for k, v in az.items() if k in AzureConfig.__annotations__}) if az else AzureConfig(),
            agent=AgentConfig(**{k: v for k, v in ag.items() if k in AgentConfig.__annotations__}) if ag else AgentConfig(),
            video=VideoConfig(**{k: v for k, v in vd.items() if k in VideoConfig.__annotations__}) if vd else VideoConfig(),
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
