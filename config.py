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
    """The clip stage runs on the GPU machine, through ComfyUI.

    LTX-2.5 is local weights; Minimax H3 and Runway Gen-4 are reached through
    ComfyUI's API nodes. So all three are workflows rather than direct REST
    calls: one connection to configure, and the vendor credentials live in
    ComfyUI where they belong rather than in this app's config file.
    """
    minimax_workflow_path: str = ""
    ltx_workflow_path: str = ""
    runway_workflow_path: str = ""


@dataclass
class AgentConfig:
    """How much rope the assistant director gets when running on its own."""
    max_steps: int = 40
    auto_lock: bool = False
    auto_install: bool = False
    variants_per_generation: int = 4
    critique_enabled: bool = True
    max_iterations_per_asset: int = 3
    # Kontext is the default because it is a different mechanism, not a better
    # setting. It encodes the reference into the sequence the model denoises, so
    # the character arrives intact; IP-Adapter hands the model a summary of the
    # reference and loses costume detail no weight can recover. Measured on the
    # Pig & Rooster reference: Kontext held the comb, wattle and eye; IP-Adapter
    # substituted a generic rooster comb at both 1.0 and 1.3.
    reference_conditioning: str = "kontext"
    # Kontext expects low guidance -- it is transforming an image it can see,
    # not inventing one from a description.
    kontext_guidance: float = 2.5
    # Kept for the fallback path. The node's own default is 1.0; anything below
    # that is weaker than the adapter's baseline, which is a strange place to
    # start when the complaint is that identity drifts.
    ipadapter_weight: float = 1.0


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
