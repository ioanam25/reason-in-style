"""The student-side feature extractor must stay identical to the teacher-side one.

`handcrafted_style_features.extract_features` is a verbatim copy of the function in
`score_scas_handcrafted_style`; if the teacher-side definition drifts, student
generations would no longer be comparable to the GMM cluster centroids.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from scripts.handcrafted_style_features import (  # noqa: E402
    DENSITY_FEATURES,
    extract_features,
)

SAMPLES = [
    "",
    "The answer is 4.",
    "Step 1: let me check this.\nWait, actually that is wrong. Let me reconsider.\n"
    "We have $x = 2$, therefore $x^2 = 4$. So the answer is \\boxed{4}?",
    "Hmm, what if we try another approach? On the other hand, let's verify: 2 + 2 = 4.\n"
    "1. first\n2. second\nIn conclusion, hence the final answer is \\boxed{4}.",
    "no punctuation at all just words running on and on without any structure whatsoever",
]


def _reference():
    ref = pytest.importorskip(
        "scripts.score_scas_handcrafted_style",
        reason="reference module needs the full training env (torch/umap)",
    )
    return ref


@pytest.mark.parametrize("text", SAMPLES)
def test_matches_reference_extractor(text):
    ref = _reference()
    assert extract_features(text) == ref.extract_features(text)


def test_density_feature_list_matches_reference():
    ref = _reference()
    assert DENSITY_FEATURES == ref.DENSITY_FEATURES


@pytest.mark.parametrize("text", SAMPLES)
def test_densities_are_length_invariant_under_duplication(text):
    """Doubling a trace leaves per-100-word densities unchanged (up to joins)."""
    if not text.strip():
        pytest.skip("empty trace has no density")
    single = extract_features(text)
    doubled = extract_features(text + "\n" + text)
    assert doubled["n_words"] >= single["n_words"]
    assert doubled["dens_backtrack"] == pytest.approx(single["dens_backtrack"], rel=0.05, abs=0.05)


def test_empty_trace_is_finite():
    feats = extract_features("")
    assert feats["n_words"] == 1.0  # guarded divisor
    assert all(v == v for v in feats.values())  # no NaNs
