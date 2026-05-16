"""Adversarial / negative phrase generation.

Two pools feed the negative training set:

1. ``GENERIC_NEGATIVE_PHRASES`` - hand-curated everyday speech that should
   never trigger any wake word.
2. ``_generate_prefix_negatives(wake_word)`` - phrases that deliberately share
   the wake word's leading phonemes. This is the critical anti-shortcut pool:
   without it the classifier learns "audio starts with /oʊk/ -> wake word"
   instead of "the full 'ok nabu' pattern -> wake word". The model needs to
   see many 'ok X' / 'oh X' negatives to be forced past the prefix.

The openwakeword library also has its own LLM-driven generator. We use this
deterministic offline generator so the trainer has no external dependency.
"""
from __future__ import annotations

import random
from pathlib import Path


# Common short phrases that can be near-confusable with various wake words.
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


# Common conversational fillers used to combine with a wake-word prefix.
# E.g. for prefix "ok": "ok so", "ok then", "ok wait", ...
_COMMON_FILLERS = [
    "so", "then", "wait", "listen", "ready", "cool", "fine", "now", "well",
    "great", "actually", "interesting", "dear", "look", "man", "no",
    "wow", "yeah", "nice", "really", "sure", "true", "good", "bye",
    "i need", "i think", "i'll", "i'm", "i guess", "i wonder",
    "let me", "let's go", "everyone", "forget it", "whatever",
    "come on", "my god", "boy", "well done", "hold on", "hang on",
    "you know", "i mean", "i was", "that is", "this is", "here we go",
    "alright", "perfect", "got it", "right",
]


# Phonetic siblings of common wake-word first-words. Used to expand the
# prefix neighbor pool. Add entries here as new wake words come up.
_PREFIX_FAMILIES = {
    "ok":      ["ok", "okay", "oh", "k", "oki"],
    "okay":    ["okay", "ok", "oh", "k"],
    "oh":      ["oh", "ok", "okay"],
    "hey":     ["hey", "hi", "yeah", "yo", "ay"],
    "hi":      ["hi", "hey", "yo"],
    "alexa":   ["alexa", "alex", "alexis", "alesa"],
    "siri":    ["siri", "syri", "siree", "cyrie"],
    "google":  ["google", "googol", "gooble"],
    "jarvis":  ["jarvis", "jarvys", "service", "jervis"],
    "computer":["computer", "computa", "compute", "commuter"],
    "nabu":    ["nabu", "naboo", "nahboo", "nah boo", "nah-boo"],
    "mycroft": ["mycroft", "microft", "micropht"],
    "rhasspy": ["rhasspy", "raspy", "rasppy"],
    "cortana": ["cortana", "cortina", "kortana"],
    "marvin":  ["marvin", "marven", "marvyn"],
}


# Common words starting with the same vowel sound. Useful as adversarials
# for wake words whose first word begins with a vowel.
_VOWEL_INITIAL_NEIGHBORS = {
    "o": [
        "only", "open", "over", "obviously", "overall", "office", "orange",
        "old", "october", "ocean", "ostrich", "opera", "option", "outside",
        "ought", "owner", "olive", "onion", "oxygen",
    ],
    "a": [
        "after", "always", "and", "any", "again", "ask", "actually",
        "answer", "amazing", "available", "almost", "alone", "along",
    ],
    "e": [
        "everyone", "everything", "easy", "either", "early", "evening",
        "elephant", "energy", "eight", "engine",
    ],
    "i": [
        "into", "instead", "important", "inside", "indeed", "it is",
        "if you", "is it", "is that",
    ],
    "u": [
        "under", "unless", "until", "upstairs", "unbelievable",
    ],
}


# Wake-word-specific common-phrase prefixes. When the wake word's first
# letter matches, all of these get included in the adversarial pool.
_O_STARTING_PHRASES = [
    "oh no", "oh wow", "oh yeah", "oh great", "oh actually", "oh interesting",
    "oh dear", "oh come on", "oh look", "oh hey", "oh man", "oh boy",
    "oh nice", "oh really", "oh sure", "oh true", "oh whatever", "oh good",
    "oh my god", "oh well", "oh fine", "oh forget it", "oh god",
    "oh that's", "oh i see", "oh that is", "oh listen",
]


# Competing wake words and rhythm-similar two-syllable openers. These are
# ALWAYS injected into the adversarial pool regardless of the trained wake
# word's prefix family, because without them the classifier never learns to
# reject the most common false-trigger class in real deployment: phrases
# that share the wake word's rhythmic shape but no phonetic content (e.g.
# "hey jarvis" for a model trained on "ok nabu"). Without these examples,
# the model can score such phrases at ~0.3-0.4 raw - thin margin against the
# 0.5 runtime trigger threshold.
_COMPETING_WAKE_WORDS = [
    # Major assistant wake words
    "hey google", "ok google", "okay google",
    "hey siri", "hi siri",
    "hey alexa", "alexa", "echo",
    "hey cortana",
    "hey computer", "computer",
    "hey portal", "hey facebook", "hey meta",
    "hey jarvis", "jarvis",
    "hey rhasspy", "rhasspy",
    "hey mycroft", "mycroft",
    "hey marvin", "marvin",
    "hey snips",
    "hey bixby", "bixby",
    "hey nest", "ok nest",
    "hey home",
    "hey assistant", "ok assistant",
    "hey buddy",
    "hey friend",
    "hey watson",
    "hey echo",
    # Common voice-command phrasing that ISN'T a wake word
    "hey there", "hi there", "hello there",
    "hey you", "hi you",
    "hey now", "hi now",
    "hey man",
    "hey listen", "hi listen",
    "hey wait",
    "hey look",
    "hey what",
    "yo wassup", "yo dude", "yo what",
    "what's up", "what is it", "what's that", "what's this", "what now",
    "wake up", "wake me", "wake me up",
    "shut up", "shut down",
    "tell me", "show me", "play me", "find me", "give me",
    "i think", "i mean", "i know", "i guess", "i wonder", "i need",
    "you know", "you see", "you got it", "you can",
    "let me", "let's go", "let's see", "let's try",
    "do it", "say it", "go on", "go ahead",
    "come on", "come here",
    "hold on", "hang on",
    "actually", "anyway", "alright", "honestly", "obviously",
    "give me a minute", "wait a sec", "wait a minute", "hold on a sec",
    "tell me about", "show me how", "find me a",
    "make me a", "play some music", "set a timer",
    "turn on", "turn off", "turn it on", "turn it off",
    "set the timer", "set the alarm",
    "what time is it", "what's the weather",
    "i love you", "i hate it",
    # Names that sound vaguely wake-word-ish (rhythm + onset)
    "hey peter", "hey patrick", "hey paul", "hey peggy",
    "hey amy", "hey andy", "hey adam", "hey alex",
    "hey betty", "hey bobby", "hey bruce", "hey ben",
    "hey carol", "hey carl", "hey casey", "hey chris",
    "hey david", "hey diana", "hey doug",
    "hey eric", "hey emma", "hey ellen",
    "hey frank", "hey fred",
    "hey george", "hey gary",
    "hey harry", "hey henry", "hey helen",
    "hey ian", "hey iris",
    "hey jack", "hey jane", "hey james", "hey jerry", "hey john",
    "hey kate", "hey kevin", "hey karl",
    "hey larry", "hey laura", "hey leo",
    "hey mary", "hey mike", "hey mark",
    "hey nick", "hey nancy", "hey nathan",
    "hey oscar", "hey oliver",
    "hey rachel", "hey ryan", "hey rob",
    "hey sarah", "hey sam", "hey sandy", "hey steve",
    "hey tom", "hey tim", "hey tony", "hey tina",
    "hey vicky", "hey vince",
    "hey walter", "hey wendy",
]


def _competing_wake_words_negatives(wake_word: str) -> list[str]:
    """Return competing-wake-word and rhythm-rival phrases minus the trained one."""
    wake_lower = wake_word.lower().strip()
    out = []
    for p in _COMPETING_WAKE_WORDS:
        if p == wake_lower:
            continue
        if " " in wake_lower and wake_lower in p:
            continue
        out.append(p)
    return out


def _prefix_family(first_word: str) -> list[str]:
    """Return phonetic siblings of a wake-word first-word (lower-cased)."""
    first_word = first_word.lower().strip()
    if first_word in _PREFIX_FAMILIES:
        return list(_PREFIX_FAMILIES[first_word])
    return [first_word]


def _generate_prefix_negatives(wake_word: str) -> list[str]:
    """Build phrases that share the wake word's leading sound.

    This is THE most important anti-shortcut signal. Without these, the
    classifier learns to fire on the first phoneme of the wake word
    (e.g. "any audio starting with 'ok' -> trigger") instead of
    requiring the full phrase.
    """
    words = wake_word.lower().split()
    if not words:
        return []

    family = _prefix_family(words[0])
    out: set[str] = set()

    # Family member alone (catches one-word utterances).
    for prefix in family:
        out.add(prefix)

    # Family member + one filler word.
    for prefix in family:
        for filler in _COMMON_FILLERS:
            out.add(f"{prefix} {filler}")

    # Family member + a second wake-word-like word (NOT the actual wake-word
    # second word). For "ok nabu" this generates "ok google", "ok siri", etc.
    # which are realistic false-trigger candidates.
    if len(words) >= 2:
        rivals = [
            "google", "siri", "alexa", "cortana", "computer",
            "assistant", "phone", "speaker", "lights", "kitchen",
            "tv", "music", "stop", "play", "pause",
        ]
        for prefix in family:
            for rival in rivals:
                if rival != words[1]:  # never accidentally generate the actual wake word
                    out.add(f"{prefix} {rival}")

    # Vowel-initial neighbors.
    first_letter = words[0][:1]
    if first_letter in _VOWEL_INITIAL_NEIGHBORS:
        for w in _VOWEL_INITIAL_NEIGHBORS[first_letter]:
            out.add(w)

    # O-starting common phrases (special-cased because they're a frequent
    # false-trigger source for 'ok'/'hey'-class wake words).
    if first_letter == "o" or "ok" in family or "oh" in family:
        for p in _O_STARTING_PHRASES:
            out.add(p)

    # Strip the actual wake word (and any concatenation containing it) so we
    # never accidentally label a positive sample as negative.
    wake_lower = wake_word.lower().strip()
    return sorted(p for p in out if p != wake_lower and wake_lower not in p)


def _phonetic_neighbors(phrase: str) -> list[str]:
    """Tiny per-phrase variation generator. Kept for back-compat with the
    classic openwakeword recipe. Prefix-family negatives are generated
    separately via ``_generate_prefix_negatives``."""
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
    return sorted(c for c in candidates if c != p)


def build_adversarial_phrases(
    wake_word: str,
    n: int,
    seed: int = 0,
    extra_phrases: list[str] | None = None,
    forbidden_phrases: list[str] | None = None,
) -> list[str]:
    """Return up to ``n`` adversarial phrases.

    Pool order of preference:
      1. Prefix-family negatives derived from the wake word (anti-shortcut).
      2. Hand-curated generic conversational phrases.
      3. Phonetic neighbors via simple swap rules.
      4. Caller-supplied extras (e.g. from the UI 'Negative phrases' field).
      5. Random pair combinations to bulk up to ``n``.
    """
    rng = random.Random(seed)
    forbidden = {
        p.lower().strip()
        for p in ([wake_word] + (forbidden_phrases or []))
        if p and p.strip()
    }

    def is_forbidden(phrase: str) -> bool:
        p = phrase.lower().strip()
        for f in forbidden:
            if f == p:
                return True
            # Multi-word positives must never appear inside a negative phrase
            # ("ok nabu, please" should not be a negative). For single-word
            # wake words, substring filtering is too aggressive: "stopwatch",
            # "stopping", and "stop sign" are useful hard negatives for
            # "stop", not mislabeled positives.
            if " " in f and f in p:
                return True
        return False

    pool: list[str] = []
    pool.extend(_generate_prefix_negatives(wake_word))
    # ALWAYS include competing wake-word + rhythm-rival negatives. Without
    # this, the model only learns to reject phrases that share its prefix
    # ("ok X") and scores rhythm-similar non-"ok" phrases ("hey jarvis")
    # at ~0.3-0.4 raw - thin margin to the 0.5 runtime trigger.
    pool.extend(_competing_wake_words_negatives(wake_word))
    pool.extend(GENERIC_NEGATIVE_PHRASES)
    pool.extend(_phonetic_neighbors(wake_word))
    if extra_phrases:
        pool.extend(extra_phrases)

    # Dedupe while preserving order (so prefix-family negatives stay near the
    # front and the head of the pool is biased toward them).
    seen: set[str] = set()
    unique: list[str] = []
    for p in pool:
        p = p.strip()
        if p and p not in seen and not is_forbidden(p):
            seen.add(p)
            unique.append(p)

    # Bulk up to n via bounded random pair combinations. This used to sample
    # pairs in a while-loop until it reached n, which can run forever when the
    # valid combination space is smaller than n (common for short words such as
    # "stop"). Shuffle the finite pair space instead.
    extra: list[str] = []
    base_for_combine = unique[:]
    pair_indices = [
        (i, j)
        for i in range(len(base_for_combine))
        for j in range(len(base_for_combine))
        if i != j
    ]
    rng.shuffle(pair_indices)
    for i, j in pair_indices:
        if len(unique) + len(extra) >= n:
            break
        a = base_for_combine[i]
        b = base_for_combine[j]
        combined = f"{a}, {b}"
        if combined not in seen and not is_forbidden(combined):
            seen.add(combined)
            extra.append(combined)
    unique.extend(extra)

    rng.shuffle(unique)
    return unique[:n]


def load_extra_phrases_file(path: Path | None) -> list[str]:
    """Optional user-supplied newline-separated phrase file."""
    if not path or not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
