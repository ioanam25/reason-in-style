"""Handcrafted style features, shared by teacher-trace and student-generation scoring.

Definitions are copied verbatim from `scripts/discovery/score_scas_handcrafted_style.py` so
that student generations land in the same feature space as the teacher-side GMM
cluster centroids. `score_scas_handcrafted_style` pulls in torch/umap via its
`eval_gemini_styles` import, which is too heavy for the streaming extractor, hence
the duplication rather than an import.

`tests/test_handcrafted_feature_parity.py` asserts the two stay identical.
"""
from __future__ import annotations

import re

import numpy as np

# --- pivot / structure lexicons (case-insensitive) -------------------------
PIVOT_PATTERNS = {
    "realization": r"\b(wait|hmm+|oh|oops|actually|hold on|i missed|my mistake|scratch that|never ?mind)\b",
    "verification": r"(let me (check|verify|confirm|make sure|re-?check|double[- ]?check|re-?examine)|to verify|to check|double[- ]?check|checking again|sanity check|let'?s verify)",
    "exploration": r"(what if|another (approach|way|method|idea)|alternativ|on the other hand|let'?s try|let me try|we could also|suppose that)",
    "integration": r"(now i see|this connects|putting (this|it) together|therefore|thus|in conclusion|to summari[sz]e|so the answer|hence|final answer)",
}
BACKTRACK_PATTERN = r"(\bwait\b|\bactually\b|but wait|on second thought|let me reconsider|scratch that|never ?mind|\binstead\b|going back)"
NUMBERED_STEP_PATTERN = r"(?mi)^\s*(?:step\s*\d+|\d+\s*[\.\):])"

_compiled = {k: re.compile(v, re.IGNORECASE) for k, v in PIVOT_PATTERNS.items()}
_bt = re.compile(BACKTRACK_PATTERN, re.IGNORECASE)
_step = re.compile(NUMBERED_STEP_PATTERN)
_sent = re.compile(r"[.!?]+")


def extract_features(text: str) -> dict[str, float]:
    text = text or ""
    n_chars = len(text)
    words = text.split()
    n_words = max(1, len(words))
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    sents = [s for s in _sent.split(text) if s.strip()]
    n_sent = max(1, len(sents))
    avg_sent_len_words = n_words / n_sent
    n_newline = text.count("\n")

    piv = {k: len(rx.findall(text)) for k, rx in _compiled.items()}
    piv_total = sum(piv.values())
    n_backtrack = len(_bt.findall(text))
    n_equals = text.count("=")
    n_dollar = text.count("$")
    n_boxed = text.count("\\boxed")
    n_steps = len(_step.findall(text))
    n_question = text.count("?")

    per100 = 100.0 / n_words

    feats = {
        # absolute / verbosity
        "n_chars": float(n_chars),
        "n_words": float(n_words),
        "n_lines": float(n_newline + 1),
        "avg_word_len": avg_word_len,
        "avg_sent_len_words": avg_sent_len_words,
        # raw pivot counts
        "piv_realization": float(piv["realization"]),
        "piv_verification": float(piv["verification"]),
        "piv_exploration": float(piv["exploration"]),
        "piv_integration": float(piv["integration"]),
        "piv_total": float(piv_total),
        "n_backtrack": float(n_backtrack),
        # raw structure
        "n_equals": float(n_equals),
        "n_dollar": float(n_dollar),
        "n_boxed": float(n_boxed),
        "n_steps": float(n_steps),
        "n_question": float(n_question),
        # length-invariant densities (per 100 words)
        "dens_realization": piv["realization"] * per100,
        "dens_verification": piv["verification"] * per100,
        "dens_exploration": piv["exploration"] * per100,
        "dens_integration": piv["integration"] * per100,
        "dens_piv_total": piv_total * per100,
        "dens_backtrack": n_backtrack * per100,
        "dens_equals": n_equals * per100,
        "dens_boxed": n_boxed * per100,
        "dens_question": n_question * per100,
        "dens_steps": n_steps * per100,
        "lines_per_100w": (n_newline + 1) * per100,
    }
    return feats


# length-invariant subset (no absolute counts)
DENSITY_FEATURES = [
    "avg_word_len",
    "avg_sent_len_words",
    "dens_realization",
    "dens_verification",
    "dens_exploration",
    "dens_integration",
    "dens_piv_total",
    "dens_backtrack",
    "dens_equals",
    "dens_boxed",
    "dens_question",
    "dens_steps",
    "lines_per_100w",
]

# compact set used for human-readable cluster profiles (matches analyze_ae_gmm_handcrafted_features)
PROFILE_FEATURES = [
    "n_words",
    "n_lines",
    "avg_sent_len_words",
    "piv_total",
    "n_backtrack",
    "n_steps",
    "n_equals",
    "n_boxed",
    "n_question",
    "dens_piv_total",
    "dens_backtrack",
    "dens_verification",
    "dens_realization",
    "dens_exploration",
    "dens_integration",
    "dens_steps",
    "dens_equals",
    "dens_boxed",
    "lines_per_100w",
]

FEATURE_NAMES = list(extract_features("x").keys())
