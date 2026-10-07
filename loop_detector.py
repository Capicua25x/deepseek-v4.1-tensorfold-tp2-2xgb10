#!/usr/bin/env python3
"""loop_detector.py — flag (and optionally cut) a reasoning loop while a reply streams.

A reasoning loop does not look like a repeated block: the text keeps looking fresh
line by line while recycling its word 8-grams. The signature is the NOVELTY of word
8-grams per fixed character window — healthy reasoning stays novel to the end, a loop
collapses below a threshold and never recovers. Three consecutive collapsed windows
sentence it.

Point it at any OpenAI-compatible /chat/completions endpoint and let it stream:

    python3 loop_detector.py --url http://HOST:PORT/v1/chat/completions \
        --body request.json --temperature 0.6            # report only
    python3 loop_detector.py ... --cut                   # close the stream on sentence

Exit code: 0 if the turn finished without a loop, 3 if a loop was sentenced.

Nothing here is specific to one engine or model: it reads the streamed `reasoning`
(``reasoning_content``) deltas and never the `content`.

Credit: the discriminator — word-n-gram novelty per window, and the ``loop_detector.py``
name — comes from the loop detector contributed to tonyd2wild's DeepSeek-V4-Flash DSpark
recipe (PR #29 there), which first told a reasoning loop from a heavy tail this way. This
is a streaming port of that idea, with word-boundary-safe windows and an optional cut.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

WORDS = re.compile(r"\w+")


class NoveltyGuard:
    """Word-n-gram novelty per window; sentence on `consecutive` dry windows."""

    def __init__(self, window: int = 4000, threshold: float = 0.02,
                 consecutive: int = 3, ngram: int = 8) -> None:
        self.window = window
        self.threshold = threshold
        self.consecutive = consecutive
        self.ngram = ngram
        self._seen: set[tuple[str, ...]] = set()
        self._pending: list[str] = []
        self._dry = 0
        self.total_chars = 0
        self.windows = 0
        self.detected = False
        self.loop_start_chars: int | None = None
        self.last_novelty: float | None = None

    def feed(self, text: str) -> bool:
        """Add streamed reasoning text; return True the first time a loop is sentenced."""
        if not text or self.detected:
            return self.detected
        self._pending.append(text)
        self.total_chars += len(text)
        buf = "".join(self._pending)
        if len(buf) < self.window:
            return False

        found = list(WORDS.finditer(buf))
        if not found:
            return False
        keep_from = found[-1].start()               # never split a word across windows
        if keep_from > 0:
            window_text, self._pending = buf[:keep_from], [buf[keep_from:]]
        else:                                        # one unbroken token: take it whole
            window_text, self._pending = buf, []
        if len(window_text) < self.window:
            self._pending = [buf]
            return False

        self.windows += 1
        words = WORDS.findall(window_text.lower())
        shingles = {tuple(words[i:i + self.ngram])
                    for i in range(max(0, len(words) - self.ngram + 1))}
        if not shingles:
            return False
        novelty = len(shingles - self._seen) / len(shingles)
        self.last_novelty = novelty
        self._seen |= shingles
        self._dry = self._dry + 1 if novelty < self.threshold else 0
        if self._dry >= self.consecutive:
            self.detected = True
            self.loop_start_chars = max(0, self.total_chars - self.window * self.consecutive)
        return self.detected


def main() -> int:
    ap = argparse.ArgumentParser(description="Flag a reasoning loop in a streaming reply.")
    ap.add_argument("--url", required=True, help="OpenAI-compatible /chat/completions URL")
    ap.add_argument("--body", required=True, help="JSON request body to send")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the body's temperature (make the loop deterministic with 0)")
    ap.add_argument("--cut", action="store_true", help="close the stream when a loop is sentenced")
    ap.add_argument("--window", type=int, default=4000)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--consecutive", type=int, default=3)
    a = ap.parse_args()

    body = json.load(open(a.body))
    if a.temperature is not None:
        body["temperature"] = a.temperature
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    req = urllib.request.Request(a.url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})

    guard = NoveltyGuard(a.window, a.threshold, a.consecutive)
    t0 = time.time()
    reason_chars = content_chars = 0
    finish = None
    try:
        res = urllib.request.urlopen(req, timeout=5400)
        for raw in res:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                rc = delta.get("reasoning") or delta.get("reasoning_content")
                if rc:
                    reason_chars += len(rc)
                    if guard.feed(rc):
                        print(f"LOOP: sentenced at {reason_chars} reasoning chars after "
                              f"{guard.windows} windows (onset ~{guard.loop_start_chars}); "
                              f"last window novelty {guard.last_novelty}", flush=True)
                        if a.cut:
                            res.close()
                            print(json.dumps({"detected": True, "cut": True,
                                              "reasoning_chars": reason_chars,
                                              "content_chars": content_chars,
                                              "seconds": round(time.time() - t0, 1)}))
                            return 3
                if delta.get("content"):
                    content_chars += len(delta["content"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    except Exception as exc:                       # noqa: BLE001 - report, don't traceback
        print(f"transport error: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(json.dumps({"detected": guard.detected, "cut": False,
                      "windows": guard.windows, "reasoning_chars": reason_chars,
                      "content_chars": content_chars, "finish_reason": finish,
                      "seconds": round(time.time() - t0, 1)}))
    return 3 if guard.detected else 0


if __name__ == "__main__":
    sys.exit(main())
