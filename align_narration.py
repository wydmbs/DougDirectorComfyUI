"""
align_narration.py — word-level Whisper alignment of narration audio against
poem beat markers, producing beats.json for the Registry tab.

Usage (on TheBeast):
    pip install openai-whisper
    python align_narration.py "Pig_and_Rooster_Mick_Update_To_Mono_28th_Nov.wav"
"""

import json
import re
import sys

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


def find_marker_start(words: list, marker_text: str, search_from_idx: int) -> tuple[int, float] | None:
    """Return the matching word index and its start time, if found."""
    marker_words = normalize(marker_text).split()
    if not marker_words:
        return None
    n = len(marker_words)
    for i in range(search_from_idx, len(words) - n + 1):
        window = [normalize(word[0]) for word in words[i:i + n]]
        matches = sum(1 for actual, expected in zip(window, marker_words) if actual == expected)
        if matches >= max(1, n - 2):
            return i, words[i][1]
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python align_narration.py <audio_file.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print("Loading Whisper model (first run downloads ~1.5GB, cached after that)...")
    import whisper

    model = whisper.load_model("medium.en")
    print(f"Transcribing {audio_path} with word-level timestamps (this can take a few minutes)...")
    result = model.transcribe(audio_path, word_timestamps=True, language="en")

    words = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            words.append((word["word"].strip(), word["start"], word["end"]))

    print(f"Transcribed {len(words)} words. Locating beat markers...")
    total_duration = words[-1][2] if words else 0.0
    starts = []
    search_from = 0
    for name, marker_text in BEAT_MARKERS:
        match = find_marker_start(words, marker_text, search_from)
        if match is None:
            print(f"  Couldn't confidently locate '{name}' — set its start_s by hand in beats.json.")
            starts.append((name, None))
            continue

        word_index, start = match
        print(f"  '{name}' starts at {start:.1f}s")
        starts.append((name, start))
        search_from = word_index + 1

    beats = []
    for i, (name, start) in enumerate(starts):
        if start is None:
            beats.append({
                "beat": name,
                "start_s": None,
                "end_s": None,
                "duration_s": None,
                "source": "whisper_alignment",
                "notes": "COULD NOT LOCATE — set manually",
            })
            continue

        next_start = next((candidate for _, candidate in starts[i + 1:] if candidate is not None), total_duration)
        beats.append({
            "beat": name,
            "start_s": round(start, 2),
            "end_s": round(next_start, 2),
            "duration_s": round(next_start - start, 2),
            "source": "whisper_alignment",
            "notes": "",
        })

    out_path = "beats.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(beats, file, indent=2)
    print(f"\nWrote {out_path} — upload it via the Registry tab's Import button.")
    unresolved = [beat["beat"] for beat in beats if beat["start_s"] is None]
    if unresolved:
        print(f"NOTE: these beats need manual start_s/end_s before importing: {unresolved}")


if __name__ == "__main__":
    main()
