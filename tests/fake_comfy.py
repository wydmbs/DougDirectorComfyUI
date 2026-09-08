"""A stand-in ComfyUI, so the doctor's deeper checks can be exercised.

The checks that matter most -- installed nodes, model files, whether a workflow
can actually hold a character's identity -- are exactly the ones that never run
when ComfyUI is absent. This serves just enough of the API to reach them.

    python tests\fake_comfy.py --port 8199 --profile complete
    python tests\fake_comfy.py --port 8199 --profile no_ipadapter

Profiles model the states a real setup passes through, so the doctor can be
tested against a machine that is half-configured rather than only against a
perfect one or an absent one.
"""

import argparse
import json
import http.server
import socketserver

BASE_NODES = {
    "CheckpointLoaderSimple": {"input": {"required": {
        "ckpt_name": [["flux1-dev-fp8.safetensors", "sd_xl_base_1.0.safetensors"]]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING"]}}},
    "KSampler": {"input": {"required": {"seed": ["INT"]}}},
    "VAEEncode": {"input": {"required": {}}},
    "LoadImage": {"input": {"required": {"image": [["example.png"]]}}},
    "SaveImage": {"input": {"required": {}}},
    "EmptyLatentImage": {"input": {"required": {}}},
}

IPADAPTER_NODES = {
    "IPAdapterAdvanced": {"input": {"required": {"weight": ["FLOAT"]}}},
    "IPAdapterModelLoader": {"input": {"required": {
        "ipadapter_file": [["ip-adapter-plus_sd15.safetensors"]]}}},
    "CLIPVisionLoader": {"input": {"required": {
        "clip_name": [["CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"]]}}},
}

# FLUX uses a different adapter entirely. A machine can have the SD/SDXL nodes
# and still be unable to condition a FLUX render, which is the trap worth
# reproducing.
IPADAPTER_FLUX_NODES = {
    "ApplyIPAdapterFlux": {"input": {"required": {"weight": ["FLOAT"]}}},
    "IPAdapterFluxLoader": {"input": {"required": {
        "ipadapter": [["instantx_flux1_dev_ip_adapter_bf16.safetensors"]]}}},
}

VIDEO_NODES = {
    "VHS_VideoCombine": {"input": {"required": {}}},
    "LTXVImgToVideo": {"input": {"required": {"num_frames": ["INT"]}}},
}

PROFILES = {
    # Everything present, FLUX adapter included: the state to call ready.
    "complete": {**BASE_NODES, **IPADAPTER_NODES, **IPADAPTER_FLUX_NODES, **VIDEO_NODES},
    # ComfyUI up, IP-Adapter never installed. The most common real gap, and the
    # one that silently ruins continuity if it goes unnoticed.
    "no_ipadapter": BASE_NODES,
    # The trap found on the real machine: SD/SDXL adapter present, FLUX one
    # absent. Everything looks installed and FLUX still cannot be conditioned.
    "sd_only": {**BASE_NODES, **IPADAPTER_NODES, **VIDEO_NODES},
    # FLUX node installed but its model file never downloaded.
    "no_flux_adapter_model": {
        **BASE_NODES, **VIDEO_NODES,
        "ApplyIPAdapterFlux": IPADAPTER_FLUX_NODES["ApplyIPAdapterFlux"],
        "IPAdapterFluxLoader": {"input": {"required": {"ipadapter": [[]]}}},
    },
    # IP-Adapter nodes installed but no model file downloaded.
    "no_model": {
        **BASE_NODES,
        "IPAdapterAdvanced": IPADAPTER_NODES["IPAdapterAdvanced"],
        "IPAdapterModelLoader": {"input": {"required": {"ipadapter_file": [[]]}}},
        "CLIPVisionLoader": {"input": {"required": {"clip_name": [[]]}}},
    },
    # No FLUX checkpoint, only an unrelated one.
    "no_flux": {
        **{k: v for k, v in BASE_NODES.items() if k != "CheckpointLoaderSimple"},
        "CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": [["sd_xl_base_1.0.safetensors"]]}}},
        **IPADAPTER_NODES, **IPADAPTER_FLUX_NODES,
    },
}

STATS = {
    "system": {"comfyui_version": "0.3.0-fake"},
    "devices": [{"name": "NVIDIA GeForce RTX 4090 (fake)",
                 "vram_total": 24 * 1024 ** 3,
                 "vram_free": 21 * 1024 ** 3}],
}


def make_handler(nodes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, payload, code=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/system_stats"):
                self._send(STATS)
            elif self.path.startswith("/object_info"):
                self._send(nodes)
            elif self.path.startswith("/queue"):
                self._send({"queue_running": [], "queue_pending": []})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            self._send({"prompt_id": "fake-prompt-id"})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--profile", default="complete", choices=sorted(PROFILES))
    args = parser.parse_args()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port),
                                make_handler(PROFILES[args.profile])) as httpd:
        print(f"fake ComfyUI [{args.profile}] on 127.0.0.1:{args.port}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
