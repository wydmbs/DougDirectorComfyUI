import asyncio
import base64
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

import main
import specialists
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
            "clare_character_brief": specialists.clare_character_brief,
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
            elif name == "clare_character_brief":
                specialists.clare_character_brief = value
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

    def test_call_sheet_cards_hide_details_until_expanded(self):
        plan = {
            "title": "Card test",
            "summary": "summary",
            "questions": [],
            "items": [{
                "kind": "BACKDROP",
                "suggested_id": "BACKDROP:dockside_loading",
                "name": "Dockside Loading",
                "description": "Period dockside for the farm-cart-to-ship transfer.",
                "continuity_note": "Storyboard V15 local review environment lock.",
            }],
        }
        html = main.call_sheet_fragment(plan)
        self.assertIn('<details class="recommendation-details">', html)
        self.assertIn("Details and amend", html)
        self.assertIn("Amend this card", html)

    def test_import_cast_opens_clare_after_successful_handoff(self):
        record = {
            "plan_id": "run-1",
            "plan": {"title": "Pig and Rooster", "items": [{"kind": "CHARACTER", "suggested_id": "CHARACTER:pig", "name": "Pig"}]},
        }
        main.call_sheets["run-1"] = record
        response = self.client.post("/clare/import", data={"plan_id": "run-1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("HX-Trigger"), "open-clare")
        self.assertIn("Cast imported", response.text)
        workspace = self.client.get("/specialists/clare/current")
        self.assertIn("Choose the character Clare develops first.", workspace.text)
        self.assertIn("Pig", workspace.text)
        selected = self.client.post("/clare/character/select", data={"character_id": "CHARACTER:pig"})
        self.assertEqual(selected.status_code, 200)
        self.assertIn("Talk to Clare about Pig.", selected.text)
        self.assertIn("Ask Clare for a character brief", selected.text)
        self.assertIn("character reference", selected.text)

    def _clare_brief(self):
        return {"identity_direction": {"core": "calm"}, "wardrobe_direction": {"cap": "tweed"}, "sheet_plan": "four views", "drift_risks": [{"risk": "toy", "prevention": "adult anatomy"}], "questions": ["Keep the cap?"], "created_at": "now"}

    def test_clare_answers_refine_brief_and_clear_trials(self):
        record = {"plan_id": "run-1", "plan": {"title": "Pig", "items": [{"kind": "CHARACTER", "suggested_id": "CHARACTER:pig", "name": "Pig"}]}}
        main.call_sheets["run-1"] = record
        self.client.post("/clare/import", data={"plan_id": "run-1"})
        self.client.post("/clare/character/select", data={"character_id": "CHARACTER:pig"})
        state = main.load_clare_development("pig_and_rooster")
        state["character_briefs"] = {"CHARACTER:pig": self._clare_brief()}
        state["character_trials"] = {"CHARACTER:pig": [{"name": "old"}]}
        main.save_clare_development("pig_and_rooster", state)
        received = []
        specialists.clare_character_brief = lambda *_args: received.append(_args[2]) or self._clare_brief()
        response = self.client.post("/clare/character/questions", data={"character_id": "CHARACTER:pig", "answer_0": "Yes, throughout."})
        self.assertEqual(response.status_code, 200)
        self.assertIn("CURRENT CHARACTER BRIEF", response.text)
        self.assertIn("Yes, throughout.", received[0])
        self.assertNotIn("CHARACTER:pig", main.load_clare_development("pig_and_rooster").get("character_trials", {}))

    def test_clare_trials_require_brief_then_prepare_four_trials(self):
        record = {"plan_id": "run-2", "plan": {"title": "Pig", "items": [{"kind": "CHARACTER", "suggested_id": "CHARACTER:pig", "name": "Pig"}]}}
        main.call_sheets["run-2"] = record
        self.client.post("/clare/import", data={"plan_id": "run-2"})
        self.client.post("/clare/character/select", data={"character_id": "CHARACTER:pig"})
        state = main.load_clare_development("pig_and_rooster")
        state["character_briefs"] = {"CHARACTER:pig": self._clare_brief()}
        main.save_clare_development("pig_and_rooster", state)
        response = self.client.post("/clare/character/trials", data={"character_id": "CHARACTER:pig"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Character trial plan prepared", response.text)
        self.assertEqual(len(main.load_clare_development("pig_and_rooster")["character_trials"]["CHARACTER:pig"]), 4)

    def test_nested_clare_brief_renders_labelled_fields(self):
        rendered = main.clare_brief_value({"flat_cap": "tweed", "palette": ["olive"]})
        self.assertIn("flat cap", rendered)
        self.assertIn("tweed", rendered)
        self.assertIn("olive", rendered)

    def test_visual_provider_payloads_use_bounded_jpeg(self):
        image_path = self.root / "reference.png"
        Image.new("RGB", (2400, 1800), "red").save(image_path)
        original_request, original_azure, original_key = harry._request, harry._azure_settings, os.environ.get("ANTHROPIC_API_KEY")
        calls = []
        harry._request = lambda url, headers, payload, **_: calls.append((url, payload)) or ({"choices": [{"message": {"content": "{}"}}]} if "openai" in url else {"content": [{"type": "text", "text": "{}"}]})
        harry._azure_settings = lambda: {"key": "key", "endpoint": "https://example.test", "deployment": "vision", "key_env": "KEY"}
        os.environ["ANTHROPIC_API_KEY"] = "key"
        try:
            harry._chat_with_images("Azure OpenAI", "system", "user", [{"path": str(image_path), "role": "shape"}])
            harry._chat_with_images("Claude", "system", "user", [{"path": str(image_path), "role": "shape"}])
        finally:
            harry._request, harry._azure_settings = original_request, original_azure
            if original_key is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = original_key
        azure_content = calls[0][1]["messages"][1]["content"]
        azure_url = next(item["image_url"]["url"] for item in azure_content if item.get("type") == "image_url")
        self.assertTrue(azure_url.startswith("data:image/jpeg;base64,"))
        self.assertLess(len(base64.b64decode(azure_url.split(",", 1)[1])), 3 * 1024 * 1024)
        claude_content = calls[1][1]["messages"][0]["content"]
        self.assertEqual(next(item for item in claude_content if item.get("type") == "image")["source"]["media_type"], "image/jpeg")

    def test_uploaded_source_already_in_editor_is_not_duplicated(self):
        harry.extract_text_document = lambda _: "Same source"
        harry.analyze_staged = lambda *_: iter(({"stage": "complete", "plan": {"title": "test", "summary": "done", "questions": [], "items": []}},))
        harry.label_script_passages = lambda *_: []
        response = self.client.post("/harry/analyze", data={"title": "test", "source": "Same source"}, files={"source_file": ("story.txt", b"Same source", "text/plain")})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Give Harry", response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
