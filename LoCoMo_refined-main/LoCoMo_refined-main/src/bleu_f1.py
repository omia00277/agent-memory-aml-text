"""Lightweight lexical metrics for LoCoMo evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def compute_bleu1(prediction: str, reference: str) -> float:
    """Compute a simple unigram BLEU score with brevity penalty."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(count, ref_counts[token]) for token, count in pred_counts.items())
    precision = overlap / len(pred_tokens)
    if precision <= 0:
        return 0.0

    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1.0 - (len(ref_tokens) / len(pred_tokens)))
    return brevity_penalty * precision


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 from bag-of-words overlap."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(count, ref_counts[token]) for token, count in pred_counts.items())
    if overlap <= 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]
