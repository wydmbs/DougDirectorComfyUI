import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
import harry_advisor as harry


class FastApiCoverageRepairTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ddc_fastapi_test_"))
        self.workspace = {
            "library": self.root / "library",
            "images": self.root / "images",
            "storyboard": self.root / "storyboard.xlsx",
        }
        self.originals = {
            "workspace": main.projects.workspace,
            "current_cfg": main.current_cfg,
            "analyze_staged": harry.analyze_staged,
            "resume_coverage": harry.resume_coverage,
            "label_script_passages": harry.label_script_passages,
            "extract_text_document": harry.extract_text_document,
        }
        main.projects.workspace = lambda _: self.workspace
        main.current_cfg = lambda: SimpleNamespace(active_project_id="pig_and_rooster", harry_provider="Fake Azure", storyboard_path=self.workspace["storyboard"], images_dir=self.workspace["images"], mock_mode=True, comfyui_url="http://127.0.0.1:8188")
        harry.set_library_dir(self.workspace["library"])
        self.client = TestClient(main.app)

    def tearDown(self):
        for name, value in self.originals.items():
            if name == "workspace":
                main.projects.workspace = value
            elif name == "current_cfg":
                main.current_cfg = value
            else:
                setattr(harry, name, value)

    def test_429_after_assets_checkpoints_and_retry_uses_coverage_only(self):
        def staged(*_):
            yield {"stage": "cast", "items": [{"kind": "CHARACTER", "name": "Pig", "suggested_id": "CHARACTER:pig"}], "source_path": "source-1"}
            yield {"stage": "world", "items": [{"kind": "BACKDROP", "name": "Farm", "suggested_id": "BACKDROP:farm"}]}
            yield {"stage": "props", "items": [{"kind": "PROP", "name": "Crate", "suggested_id": "PROP:crate"}]}
            raise harry.HarryRateLimitError("simulated Azure 429")

        harry.analyze_staged = staged
        response = self.client.post("/harry/analyze", data={"title": "Harry test", "source": "Pig sees a crate.", "era": "England"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Coverage paused, not lost", response.text)
        checkpoint = main.load_coverage_checkpoint("pig_and_rooster")
        self.assertEqual([item["name"] for item in checkpoint["characters"]], ["Pig"])
        self.assertEqual([item["name"] for item in checkpoint["backdrops"]], ["Farm"])
        self.assertEqual([item["name"] for item in checkpoint["props"]], ["Crate"])
        self.assertEqual(checkpoint["completed_stages"], ["cast", "world", "props"])
        self.assertIn("Retry Coverage", self.client.get("/fragments/dashboard").text)

        calls = []
        def resume(provider, saved, progress):
            calls.append((provider, saved))
            return {"title": saved["title"], "summary": "done", "questions": [], "items": saved["characters"] + saved["backdrops"] + saved["props"] + [{"kind": "SHOT", "name": "Opening", "camera_direction": "wide", "lighting_direction": "dawn"}]}

        harry.resume_coverage = resume
        harry.label_script_passages = lambda *_: []
        response = self.client.post("/harry/retry-coverage")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Opening", response.text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "Fake Azure")
        self.assertIsNone(main.load_coverage_checkpoint("pig_and_rooster"))

    def test_partial_checkpoint_does_not_offer_or_run_coverage_retry(self):
        main.save_coverage_checkpoint("pig_and_rooster", {"title": "test", "source_text": "text", "characters": [{"name": "Pig"}], "backdrops": [], "props": [], "completed_stages": ["cast"]})
        self.assertNotIn("Retry Coverage", self.client.get("/fragments/dashboard").text)
        response = self.client.post("/harry/retry-coverage")
        self.assertIn("cannot resume coverage", response.text)

    def test_extract_marks_only_successful_text_for_file_clear(self):
        harry.extract_text_document = lambda _: "Extracted story text"
        response = self.client.post("/harry/extract", files={"source_file": ("story.txt", b"story", "text/plain")})
        self.assertEqual(response.headers.get("HX-Trigger"), "source-extracted")
        self.assertIn("Extracted story text", response.text)
        response = self.client.post("/harry/extract", files={"source_file": ("story.exe", b"x", "application/octet-stream")})
        self.assertIsNone(response.headers.get("HX-Trigger"))
        self.assertIn("can read", response.text)

    def test_uploaded_source_already_in_editor_is_not_duplicated(self):
        harry.extract_text_document = lambda _: "Same source"
        harry.analyze_staged = lambda *_: iter(({"stage": "complete", "plan": {"title": "test", "summary": "done", "questions": [], "items": []}},))
        harry.label_script_passages = lambda *_: []
        response = self.client.post("/harry/analyze", data={"title": "test", "source": "Same source"}, files={"source_file": ("story.txt", b"Same source", "text/plain")})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Give Harry", response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
