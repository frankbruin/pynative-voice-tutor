"""
compare_asr_prompt.py
---------------------
Small experiment: does the vocabulary hint (config.ASR_PROMPT) improve how
well Whisper recognises Python terms?

For every recording in asr_test/recordings/ it transcribes the audio TWICE
(without and with the prompt) and compares both to the expected sentence in
asr_test/sentences.csv, using the Word Error Rate (WER):
    WER = (substituted + deleted + inserted words) / words in expected sentence
Lower is better; 0.0 means a perfect transcription.

Run from the project folder (with the venv active):
    python asr_test/compare_asr_prompt.py
Results are printed and saved to asr_test/results.csv.
"""

from __future__ import annotations  # allows "Path | None" on Python 3.9

import csv
import re
import sys
from pathlib import Path

from faster_whisper import WhisperModel

# Make the project folder importable, so we can reuse config.py and the
# audio decoder from asr_utils.py (same settings as the real app).
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
import config  # noqa: E402
from asr_utils import _decode_to_numpy_16k  # noqa: E402

TEST_DIR = PROJECT_DIR / "asr_test"
RECORDINGS_DIR = TEST_DIR / "recordings"


def normalise(text: str) -> list[str]:
    """Lowercase, drop punctuation (except - and _), split into words, so that
    'Tuple,' and 'tuple' count as the same word."""
    text = text.lower().replace("*", "")
    return re.sub(r"[^\w\s-]", " ", text).split()


def word_error_rate(expected: str, heard: str) -> float:
    """Classic WER via edit distance on words (dynamic programming)."""
    ref, hyp = normalise(expected), normalise(heard)
    if not ref:
        return 0.0
    # dist[i][j] = edits needed to turn ref[:i] into hyp[:j]
    dist = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dist[i][0] = i
    for j in range(len(hyp) + 1):
        dist[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j] + 1,         # deletion
                             dist[i][j - 1] + 1,         # insertion
                             dist[i - 1][j - 1] + cost)  # substitution
    return dist[len(ref)][len(hyp)] / len(ref)


def find_recording(file_id: str) -> Path | None:
    """Finds e.g. recordings/01.m4a or 01.wav or 01.mp3 (any audio format)."""
    matches = sorted(RECORDINGS_DIR.glob(f"{file_id}.*"))
    return matches[0] if matches else None


def transcribe(model: WhisperModel, audio, prompt: str | None) -> str:
    segments, _info = model.transcribe(
        audio, language=config.WHISPER_LANGUAGE, beam_size=1, initial_prompt=prompt
    )
    return " ".join(s.text for s in segments).strip()


def main():
    print(f"Loading Whisper model '{config.WHISPER_MODEL_SIZE}'...")
    model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    with open(TEST_DIR / "sentences.csv", newline="", encoding="utf-8") as f:
        sentences = list(csv.DictReader(f))

    results = []
    for row in sentences:
        path = find_recording(row["file"])
        if path is None:
            print(f"[skip] no recording found for {row['file']}")
            continue
        audio = _decode_to_numpy_16k(path.read_bytes())
        without = transcribe(model, audio, None)
        with_prompt = transcribe(model, audio, config.ASR_PROMPT)
        result = {
            "file": path.name,
            "expected": row["expected"],
            "without_prompt": without,
            "wer_without": round(word_error_rate(row["expected"], without), 3),
            "with_prompt": with_prompt,
            "wer_with": round(word_error_rate(row["expected"], with_prompt), 3),
        }
        results.append(result)
        print(f"\n{path.name}")
        print(f"  expected : {result['expected']}")
        print(f"  without  : {without}   (WER {result['wer_without']})")
        print(f"  with     : {with_prompt}   (WER {result['wer_with']})")

    if not results:
        print("\nNo recordings found - put them in asr_test/recordings/ (01.m4a, 02.m4a, ...).")
        return

    avg_without = sum(r["wer_without"] for r in results) / len(results)
    avg_with = sum(r["wer_with"] for r in results) / len(results)
    print(f"\nAverage WER over {len(results)} recordings:")
    print(f"  without prompt: {avg_without:.3f}")
    print(f"  with prompt   : {avg_with:.3f}")

    with open(TEST_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {TEST_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
