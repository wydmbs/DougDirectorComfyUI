import tempfile
import unittest
from pathlib import Path

import clare_pipeline as clare
import storyboard_store as store


class ClarePipelineTests(unittest.TestCase):
    def setUp(self):
        self.path = str(Path(tempfile.mkdtemp()) / "storyboard.xlsx")
        clare.cache_identity_locks(self.path)
        self.shot = {
            "suggested_id": "4.4",
            "description": "Bertie and Pig endure a storm.",
            "camera_direction": "full-body wide shot",
            "positive_prompt": clare.THELWELL_STYLE_BLOCK + " storm at sea",
        }

    def block(self, name="Pig", **overrides):
        lock = clare.IDENTITY_LOCKS[name]
        block = {
            "name": name,
            "trigger_phrase": lock["trigger_phrase"],
            "lora_file": lock["lora_file"],
            "lora_strength": lock["lora_strength"],
            "style_block_required": True,
            "wardrobe": {"base": lock["base_wardrobe"], "variant": "storm-soaked transformation of locked base"},
            "pose_state": "upright" if name == "Bertie" else "n/a",
            "framing": "full-body",
            "out_of_frame_check": "pass",
            "human_visibility_tier": "n/a",
            "children_appropriate": "pass",
        }
        block.update(overrides)
        return block

    def test_identity_locks_are_cached_per_character(self):
        self.assertIn("pig_lora_final.safetensors", store.get_character(self.path, "CHARACTER:pig")["identity_lock_json"])
        self.assertIn("rooster_lora_p3.safetensors", store.get_character(self.path, "CHARACTER:bertie")["identity_lock_json"])

    def test_approved_shot_persists_resumable_result(self):
        result = clare.validate_and_store(self.path, self.shot, [self.block(), self.block("Bertie")])
        self.assertEqual(result["status"], "approved")
        entry = store.get_entry(self.path, "4.4")
        self.assertEqual(entry["clare_status"], "approved")
        self.assertEqual(len(clare.load_result(entry)), 2)
        self.assertTrue(clare.is_downstream_eligible(entry))

    def test_missing_style_block_rejects(self):
        self.shot["positive_prompt"] = "storm at sea"
        self.assertEqual(clare.validate_shot(self.shot, [self.block()])["status"], "rejected")

    def test_wardrobe_redesign_rejects(self):
        block = self.block(wardrobe={"base": "locked base", "variant": "entirely new replacement outfit redesign"})
        self.assertEqual(clare.validate_shot(self.shot, [block])["status"], "rejected")

    def test_close_up_requires_out_of_frame_pass(self):
        block = self.block(framing="close-up", out_of_frame_check="fail")
        self.assertEqual(clare.validate_shot(self.shot, [block])["status"], "rejected")

    def test_human_requires_visibility_tier(self):
        self.shot["description"] = "A human loads the crate."
        self.assertEqual(clare.validate_shot(self.shot, [self.block()])["status"], "rejected")

    def test_rejected_shot_is_not_downstream_eligible(self):
        self.shot["positive_prompt"] = "storm at sea"
        result = clare.validate_shot(self.shot, [self.block()])
        store.set_clare_result(self.path, "4.4", result["status"], result["character_blocks"], result["notes"], result["checked_at"])
        self.assertFalse(clare.is_downstream_eligible(store.get_entry(self.path, "4.4")))


if __name__ == "__main__":
    unittest.main()
