"""
Extract interpretable reasoning-style features from raw traces.

Each function takes a trace string and returns a scalar feature value.
These features are used by probe_z.py to test what z(s) encodes.
"""

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

_STEP_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:"
    r"step\s+\d+"
    r"|(?:first|second|third|fourth|fifth|next|then|finally|therefore|thus|hence|so)\b"
    r"|\d+[\.\)]\s"
    r")",
    re.IGNORECASE,
)

_BACKTRACK_MARKERS = re.compile(
    r"\b(?:wait|actually|no[,.]|let me (?:re(?:consider|think|do|start|try)|go back|try (?:a different|another))"
    r"|on second thought|that(?:'s| is) (?:wrong|incorrect|not right)"
    r"|I (?:made a mistake|was wrong|need to (?:reconsider|rethink))"
    r"|scratch that|never ?mind|hmm|oops"
    r"|that doesn(?:'t| not) (?:work|seem right|look right|make sense))\b",
    re.IGNORECASE,
)

_METACOGNITIVE_MARKERS = re.compile(
    r"\b(?:let me (?:think|check|verify|consider|see|reason|analyze|calculate|work)"
    r"|I (?:need to|should|will|can|notice|observe|see that|think|believe|recall)"
    r"|(?:let's|let us) (?:think|check|verify|see|try|consider|reason|start)"
    r"|this (?:means|implies|suggests|gives|shows|tells)"
    r"|we (?:need to|should|can|know|have|see|get|find|observe)"
    r"|to (?:summarize|conclude|verify|check|find|solve|determine))\b",
    re.IGNORECASE,
)

_VERIFICATION_MARKERS = re.compile(
    r"\b(?:let me (?:check|verify|confirm|double[- ]check|make sure|validate)"
    r"|(?:to |let's |let us )(?:check|verify|confirm|validate)"
    r"|checking|verif(?:ying|ication)|sanity check"
    r"|does this (?:make sense|check out|work)"
    r"|plugging (?:back )?in|substitut(?:ing|e) back"
    r"|indeed|as expected|which (?:checks out|confirms|matches|is correct))\b",
    re.IGNORECASE,
)

_MATH_NOTATION = re.compile(
    r"(?:\$[^$]+\$"  # inline LaTeX
    r"|\\(?:frac|sqrt|int|sum|prod|lim|log|ln|sin|cos|tan|binom|begin|end)\b"
    r"|[=<>≤≥≠±×÷∈∉⊂⊃∪∩∀∃∞→←↔⇒⇐⇔]"
    r"|\b\d+\s*[+\-*/^]\s*\d+"  # arithmetic expressions
    r")"
)

_VARIABLE_ASSIGNMENT = re.compile(
    r"\b(?:let|set|define|denote)\s+\w+\s*=",
    re.IGNORECASE,
)

_ALTERNATIVE_MARKERS = re.compile(
    r"\b(?:alternativ(?:ely|e)|another (?:approach|way|method|strategy)"
    r"|(?:we )?could (?:also|instead)|instead[,.]"
    r"|approach \d|method \d|(?:case|option) \d"
    r"|on the other hand|or (?:we could|equivalently))\b",
    re.IGNORECASE,
)

_CODE_BLOCK = re.compile(r"```[\s\S]*?```")

_FILLER_PATTERNS = re.compile(
    r"\b(?:okay|ok|alright|so|well|now|right|umm?|hmm+|uh|ah|anyway|basically"
    r"|you know|I mean|like I said)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def trace_length(trace: str) -> int:
    """Raw character length."""
    return len(trace)


def word_count(trace: str) -> int:
    return len(trace.split())


def step_count(trace: str) -> int:
    """Number of explicit reasoning step markers."""
    return len(_STEP_MARKERS.findall(trace))


def backtrack_count(trace: str) -> int:
    """Number of self-correction / backtracking markers."""
    return len(_BACKTRACK_MARKERS.findall(trace))


def has_backtracking(trace: str) -> int:
    """Binary: does the trace contain any backtracking?"""
    return int(backtrack_count(trace) > 0)


def metacognition_count(trace: str) -> int:
    """Number of metacognitive commentary markers."""
    return len(_METACOGNITIVE_MARKERS.findall(trace))


def metacognition_density(trace: str) -> float:
    """Metacognitive markers per 1000 words."""
    wc = word_count(trace)
    if wc == 0:
        return 0.0
    return metacognition_count(trace) / wc * 1000


def verification_count(trace: str) -> int:
    """Number of verification/checking subroutines."""
    return len(_VERIFICATION_MARKERS.findall(trace))


def has_verification(trace: str) -> int:
    """Binary: does the trace contain verification steps?"""
    return int(verification_count(trace) > 0)


def math_notation_density(trace: str) -> float:
    """Math notation occurrences per 1000 characters."""
    n = len(trace)
    if n == 0:
        return 0.0
    return len(_MATH_NOTATION.findall(trace)) / n * 1000


def variable_assignment_count(trace: str) -> int:
    """Number of explicit variable assignments (let x = ...)."""
    return len(_VARIABLE_ASSIGNMENT.findall(trace))


def verbosity_ratio(trace: str) -> float:
    """Words per reasoning step. High = verbose, low = concise."""
    steps = max(step_count(trace), 1)
    return word_count(trace) / steps


def alternative_count(trace: str) -> int:
    """Number of times an alternative approach is mentioned."""
    return len(_ALTERNATIVE_MARKERS.findall(trace))


def exploration_breadth(trace: str) -> int:
    """Number of distinct approaches explored (alternative markers + 1 for the initial approach)."""
    alts = alternative_count(trace)
    return alts + 1 if alts > 0 else 1


def paragraph_count(trace: str) -> int:
    """Number of paragraphs (separated by double newlines)."""
    paras = re.split(r"\n\s*\n", trace.strip())
    return len([p for p in paras if p.strip()])


def newline_density(trace: str) -> float:
    """Newlines per 1000 characters (structural formatting indicator)."""
    n = len(trace)
    if n == 0:
        return 0.0
    return trace.count("\n") / n * 1000


def code_block_count(trace: str) -> int:
    """Number of fenced code blocks."""
    return len(_CODE_BLOCK.findall(trace))


def has_code(trace: str) -> int:
    """Binary: does the trace contain code blocks?"""
    return int(code_block_count(trace) > 0)


def filler_density(trace: str) -> float:
    """Filler words per 1000 words."""
    wc = word_count(trace)
    if wc == 0:
        return 0.0
    return len(_FILLER_PATTERNS.findall(trace)) / wc * 1000


def backtrack_density(trace: str) -> float:
    """Backtracking markers per 1000 words."""
    wc = word_count(trace)
    if wc == 0:
        return 0.0
    return backtrack_count(trace) / wc * 1000


def verification_density(trace: str) -> float:
    """Verification markers per 1000 words."""
    wc = word_count(trace)
    if wc == 0:
        return 0.0
    return verification_count(trace) / wc * 1000


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_FEATURES = {
    "trace_length": trace_length,
    "word_count": word_count,
    "step_count": step_count,
    "backtrack_count": backtrack_count,
    "has_backtracking": has_backtracking,
    "metacognition_count": metacognition_count,
    "metacognition_density": metacognition_density,
    "verification_count": verification_count,
    "has_verification": has_verification,
    "math_notation_density": math_notation_density,
    "variable_assignment_count": variable_assignment_count,
    "verbosity_ratio": verbosity_ratio,
    "alternative_count": alternative_count,
    "exploration_breadth": exploration_breadth,
    "paragraph_count": paragraph_count,
    "newline_density": newline_density,
    "code_block_count": code_block_count,
    "has_code": has_code,
    "filler_density": filler_density,
    "backtrack_density": backtrack_density,
    "verification_density": verification_density,
}


def extract_all_features(trace: str) -> dict[str, float]:
    """Extract all features from a single trace."""
    return {name: float(fn(trace)) for name, fn in ALL_FEATURES.items()}
