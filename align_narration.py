"""
align_narration.py — word-level forced alignment of the narration audio
against the poem text, producing a beats.json the Registry tab's "Import"
button can read directly.

WHY THIS RUNS ON THEBEAST, NOT IN A CLOUD SESSION:
Whisper's model weights are hosted on domains a cloud coding session's
network typically can't reach (huggingface.co, openaipublic's CDN), and the
model file itself (1-3GB+) may not fit in a cloud session's disk quota.
TheBeast has both a real internet connection and plenty of disk — this is
a one-time, ~5 minute run.

USAGE (on TheBeast):
    pip install openai-whisper
    python align_narration.py "Pig_and_Rooster_Mick_Update_To_Mono_28th_Nov.wav"

This produces beats.json in the same folder. Upload that file to the
Registry tab's "Import" button in the Director Harness.

HOW IT WORKS:
1. Transcribes the audio with word-level timestamps (Whisper's
   word_timestamps=True — not a separate forced-aligner, but Whisper's own
   attention-based word timing, which is good enough for beat-level
   granularity; a dedicated aligner like WhisperX would give tighter
   per-word timing if that level of precision is ever needed).
2. Matches the transcribed words against BEAT_MARKERS below (the same
   section boundaries as the poem text) to find where each beat starts.
3. Writes out one row per beat: start_s, end_s, duration_s.

If the transcription doesn't cleanly match a beat marker (mishears,
mumbled words, music under the vocals), it prints a warning for that beat
so you can manually adjust the corresponding entry in beats.json before
importing — this is meant to get you 90% of the way, not to be blindly
trusted for exact frame-accurate cuts.
"""

import json
import re
import sys

# The same beat boundaries as the poem text, in order. Each is the first
# few distinctive words of that beat, used to find where it starts in the
# transcript. Keep these in sync with the poem if the lyrics ever change.
BEAT_MARKERS = [
    ("The opening", "here is a tale of safety at sea"),
    ("The ship", "as dawn rays opens we are separate apart"),
    ("The storm", "thunder strikes hard after"),
    ("Moved to the sea", "our crates are now floating"),
    ("At Sea", "riding and bowling clinging and rolling"),
    ("The Journey Continues", "five days of shining six of the dark"),
    ("Landing of a shore", "the days become nights the nights become days"),
    ("The Shore", "we rode the storm together as strangers"),
    ("Closing", "many time this happened has happened"),
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def find_marker_start(words: list, marker_text: str, search_from_idx: int) -> float | None:
    """words: list of (word_text, start_s, end_s) tuples from Whisper, in
    order. Finds the first index >= search_from_idx where the next few
    words match marker_text, returns that word's start time."""
    marker_words = normalize(marker_text).split()
    if not marker_words:
        return None
    n = len(marker_words)
    for i in range(search_from_idx, len(words) - n + 1):
        window = [normalize(w[0]) for w in words[i:i + n]]
        # allow one or two word misses (ASR isn't perfect) — require most words to match
        matches = sum(1 for a, b in zip(window, marker_words) if a == b)
        if matches >= max(1, n - 2):
            return words[i][1]
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python align_narration.py <audio_file.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print("Loading Whisper model (first run downloads ~1.5GB, cached after that)...")
    import whisper  # local import so --help works without the dependency installed
    model = whisper.load_model("medium.en")

    print(f"Transcribing {audio_path} with word-level timestamps (this can take a few minutes)...")
    result = model.transcribe(audio_path, word_timestamps=True, language="en")

    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append((w["word"].strip(), w["start"], w["end"]))

    print(f"Transcribed {len(words)} words. Locating beat markers...")

    total_duration = words[-1][2] if words else 0.0
    starts = []
    search_from = 0
    for name, marker_text in BEAT_MARKERS:
        start = find_marker_start(words, marker_text, search_from)
        if start is None:
            print(f"  ⚠️  Couldn't confidently locate '{name}' — you'll need to set its "
                  f"start_s by hand in beats.json.")
            starts.append((name, None))
        else:
            print(f"  ✓ '{name}' starts at {start:.1f}s")
            search_from = max(search_from, int(len(words) * (start / total_duration))) if total_duration else search_from
        starts.append((name, start)) if start is not None else None

    # Build beats with end_s = next beat's start_s (or total duration for the last one)
    beats = []
    resolved_starts = [s for _, s in starts if s is not None]
    for i, (name, start) in enumerate(starts):
        if start is None:
            beats.append({"beat": name, "start_s": None, "end_s": None, "duration_s": None,
                          "source": "whisper_alignment", "notes": "COULD NOT LOCATE — set manually"})
            continue
        next_start = None
        for _, s2 in starts[i + 1:]:
            if s2 is not None:
                next_start = s2
                break
        end = next_start if next_start is not None else total_duration
        beats.append({
            "beat": name, "start_s": round(start, 2), "end_s": round(end, 2),
            "duration_s": round(end - start, 2), "source": "whisper_alignment", "notes": "",
        })

    out_path = "beats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(beats, f, indent=2)
    print(f"\nWrote {out_path} — upload it via the Registry tab's Import button.")
    unresolved = [b["beat"] for b in beats if b["start_s"] is None]
    if unresolved:
        print(f"NOTE: these beats need manual start_s/end_s before importing: {unresolved}")


if __name__ == "__main__":
    main()
