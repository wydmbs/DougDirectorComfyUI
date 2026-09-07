"""
prepare_ui_assets.py — run this ONCE, locally, after generating the UI
graphics batch (see asset_manifest.json / INVENTORY.md).

WHY THIS EXISTS: the generated assets are 1024x1024+ source art, but the
app actually displays them small (a tab-header icon at ~28px, a favicon at
16-64px). Dropping the raw 300-700KB source files straight into the app
would work but is wasteful and slow to load repeatedly. This script
resizes each asset to the size it's actually displayed at, generates a
proper multi-size .ico favicon, and copies everything into the repo's
tracked assets/ui/ folder with the exact filenames app.py expects.

USAGE:
    cd "C:\\Users\\dougl\\DougDirectorComfyui\\DougDirectorComfyUI"
    python prepare_ui_assets.py "C:\\Users\\dmccutch\\OneDrive\\Pig and Rooster Redo\\App images"

(The source folder argument is wherever asset_manifest.json's assets
actually live — adjust the path to match your machine.)

Requires Pillow (already in requirements.txt).
"""

import json
import os
import sys

from PIL import Image

# Display size each asset is actually used at in the app. Icons render as
# small header/tab touches (~28-40px in the UI) but are kept at 2x for
# retina displays; banners are capped at a sane max width for a local app.
TARGET_SIZES = {
    "app_icon": (256, 256),           # header logo + favicon source
    "tab_icon_harry": (128, 128),
    "tab_icon_build": (128, 128),
    "tab_icon_registry": (128, 128),
    "reward_seal_print_the_take": (160, 160),
    "welcome_hero_banner": (1400, None),   # cap width, preserve aspect ratio
    "celebration_wrap_banner": (900, None),
    "registry_empty_state": (1200, None),
    "favicon_source": None,  # handled separately — generates favicon.ico
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ui")


def resize_keep_aspect(img: Image.Image, target_w: int, target_h) -> Image.Image:
    if target_h is None:
        ratio = target_w / img.width
        target_h = round(img.height * ratio)
    return img.resize((target_w, target_h), Image.LANCZOS)


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_ui_assets.py <path to folder containing the generated PNGs>")
        sys.exit(1)

    source_dir = sys.argv[1]
    manifest_path = os.path.join(source_dir, "asset_manifest.json")
    if not os.path.exists(manifest_path):
        # allow running from the repo root with the manifest already copied alongside this script
        manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Couldn't find asset_manifest.json in {source_dir} or next to this script. "
              "Copy it there, or pass the correct source folder.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    for asset in manifest["assets"]:
        asset_id = asset["id"]
        src_path = os.path.join(source_dir, asset["filename"])
        if not os.path.exists(src_path):
            print(f"⚠️  SKIP {asset_id}: source file not found at {src_path}")
            continue

        img = Image.open(src_path).convert("RGBA")

        if asset_id == "favicon_source":
            # Generate a real multi-size .ico from the ultra-simple two-tone source
            ico_path = os.path.join(OUTPUT_DIR, "favicon.ico")
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
            img.save(ico_path, format="ICO", sizes=sizes)
            print(f"✅ {asset_id} -> favicon.ico ({', '.join(f'{w}x{h}' for w, h in sizes)})")
            continue

        target = TARGET_SIZES.get(asset_id)
        if target:
            img = resize_keep_aspect(img, target[0], target[1])

        out_name = f"{asset_id}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        img.save(out_path, format="PNG", optimize=True)
        print(f"✅ {asset_id} -> {out_name} ({img.width}x{img.height}, {os.path.getsize(out_path):,} bytes)")

    print(f"\nDone. Files are in {OUTPUT_DIR} — this folder is tracked in git, "
          "so 'git add assets/ui' will pick them up for commit.")


if __name__ == "__main__":
    main()
