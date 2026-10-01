"""Extract and grade \\boxed{} math answers for MATH-style benchmarks."""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from fractions import Fraction

import sympy as sp


@contextmanager
def _sympy_time_limit(seconds: float):
    """Abort sympy work that exceeds *seconds* (Unix main thread only)."""
    if seconds <= 0:
        yield
        return

    def _handler(_signum, _frame):
        raise TimeoutError("sympy grading timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

BOXED_RE = re.compile(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}")


def parse_boxed_answer(text: str) -> str:
    key = "\\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return ""
    i = idx + len(key)
    depth = 1
    start = i
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return ""


def _strip_math_wrappers(text: str) -> str:
    s = str(text).strip()
    s = s.replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace("\\,", "").replace("\\!", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\dfrac", "\\frac")
    return s


def _latex_frac_to_sympy(s: str) -> str:
    while True:
        new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
        if new == s:
            break
        s = new
    return s


def _latex_to_sympy_text(s: str) -> str:
    s = _strip_math_wrappers(s)
    s = _latex_frac_to_sympy(s)
    s = re.sub(r"(\d)\\sqrt\{([^{}]+)\}", r"\1*sqrt(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    return s


def _split_tuple(s: str) -> list[str] | None:
    s = _strip_math_wrappers(s)
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1]
    parts, cur, depth = [], [], 0
    for ch in inner:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _try_sympy_equal(a: str, b: str, *, timeout_sec: float) -> bool:
    if not a or not b:
        return False
    a_parts = _split_tuple(a)
    b_parts = _split_tuple(b)
    if a_parts is not None and b_parts is not None:
        if len(a_parts) != len(b_parts):
            return False
        return all(_try_sympy_equal(x, y, timeout_sec=timeout_sec) for x, y in zip(a_parts, b_parts))

    # Avoid ea.equals(eb) — it can hang indefinitely on hard expressions.
    for text in (_latex_to_sympy_text(a), _strip_math_wrappers(a)):
        for other in (_latex_to_sympy_text(b), _strip_math_wrappers(b)):
            try:
                with _sympy_time_limit(timeout_sec):
                    ea = sp.sympify(text)
                    eb = sp.sympify(other)
                    if sp.simplify(ea - eb) == 0:
                        return True
            except (TimeoutError, Exception):
                continue
    return False


def _try_numeric_equal(a: str, b: str, *, timeout_sec: float, tol: float = 1e-6) -> bool:
    for text in (_latex_to_sympy_text(a), _strip_math_wrappers(a)):
        for other in (_latex_to_sympy_text(b), _strip_math_wrappers(b)):
            try:
                with _sympy_time_limit(timeout_sec):
                    fa = float(sp.N(sp.sympify(text)))
                    fb = float(sp.N(sp.sympify(other)))
                if abs(fa - fb) <= tol:
                    return True
            except (TimeoutError, Exception):
                try:
                    fa = float(Fraction(text))
                    fb = float(Fraction(other))
                    if abs(fa - fb) <= tol:
                        return True
                except Exception:
                    continue
    return False


def answers_match(prediction: str, gold: str, *, timeout_sec: float = 2.0) -> bool:
    pred = parse_boxed_answer(prediction) or prediction.strip()
    gold = gold.strip()
    if not pred or not gold:
        return False

    pred_s = _strip_math_wrappers(pred)
    gold_s = _strip_math_wrappers(gold)
    if pred_s == gold_s:
        return True

    pred_parts = _split_tuple(pred)
    gold_parts = _split_tuple(gold)
    if pred_parts is not None and gold_parts is not None:
        if len(pred_parts) == len(gold_parts):
            if all(_strip_math_wrappers(x) == _strip_math_wrappers(y) for x, y in zip(pred_parts, gold_parts)):
                return True

    per_call = max(0.05, timeout_sec / 4.0)
    try:
        if _try_sympy_equal(pred, gold, timeout_sec=per_call):
            return True
        if _try_numeric_equal(pred, gold, timeout_sec=per_call):
            return True
    except Exception:
        return False
    return False
