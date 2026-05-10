"""Adversarial / negative phrase generation.

The openwakeword library has its own `generate_adversarial_texts` that uses an
LLM. We provide a lightweight, offline alternative: phonetic neighbors of the
wake word + generic conversational phrases. This is the "hard negatives" pool
synthesized via Piper alongside the positives.
"""
from __future__ import annotations

import random
from pathlib import Path

# Common short phrases that can be near-confusable with various wake words.
# Curated to exercise the FCN's discriminator (hey there, hi everyone, hello, etc).
GENERIC_NEGATIVE_PHRASES = [
    "hey there",
    "hello everyone",
    "hi how are you",
    "hello world",
    "hey can you hear me",
    "okay computer",
    "good morning",
    "good afternoon",
    "what time is it",
    "tell me a joke",
    "play some music",
    "turn on the lights",
    "set a timer for five minutes",
    "what's the weather",
    "i need help",
    "thank you very much",
    "no problem",
    "i don't understand",
    "let me think",
    "wait a second",
    "give me a minute",
    "are you serious",
    "that's amazing",
    "i'm not sure",
    "tell me more",
    "stop the music",
    "pause the video",
    "skip this song",
    "increase the volume",
    "decrease the volume",
    "what did you say",
    "say that again",
    "could you repeat that",
    "okay then",
    "i guess so",
    "maybe later",
    "see you tomorrow",
    "good night",
    "have a great day",
    "i'll be right back",
    "where are you going",
    "did you eat already",
    "can you call me",
    "send a text message",
    "open the door",
    "close the window",
    "what was i saying",
    "i need to think",
    "let's get started",
    "alright let's go",
]


def _phonetic_neighbors(phrase: str) -> list[str]:
    """Tiny phonetic variation generator: swap voiced/unvoiced, drop H, etc."""
    p = phrase.lower().strip()
    candidates = {p}
    swaps = [
        ("hey", "ay"),
        ("hey", "okay"),
        ("hey", "hi"),
        ("hi", "hey"),
        ("hi", "ay"),
        ("ok", "okay"),
        ("okay", "ok"),
        ("computer", "compuper"),
    ]
    for a, b in swaps:
        if a in p:
            candidates.add(p.replace(a, b))
    return [c for c in candidates if c != p]


def build_adversarial_phrases(
    wake_word: str,
    n: int,
    seed: int = 0,
    extra_phrases: list[str] | None = None,
) -> list[str]:
    """Return up to `n` adversarial phrases."""
    rng = random.Random(seed)
    pool: list[str] = []
    pool.extend(GENERIC_NEGATIVE_PHRASES)
    pool.extend(_phonetic_neighbors(wake_word))
    if extra_phrases:
        pool.extend(extra_phrases)

    # Combine pairs to get more variety up to n.
    extra: list[str] = []
    while len(pool) + len(extra) < n:
        a, b = rng.sample(pool, 2)
        extra.append(f"{a}, {b}")
    pool.extend(extra)
    rng.shuffle(pool)
    return pool[:n]


def load_extra_phrases_file(path: Path | None) -> list[str]:
    """Optional user-supplied newline-separated phrase file."""
    if not path or not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
