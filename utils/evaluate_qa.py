#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
Functions for computing QA evaluation metrics.

We adapt the functions from the official SQuAD (Rajpurkar et al. '18) evaluation script:
https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
"""

import collections
import re
import string
from typing import List


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Normalize string and split string into tokens."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Compute the Exact Match score."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1_from_tokens(gold_toks: List[str], pred_toks: List[str]) -> float:
    """Compute the F1 score from tokenized gold answer and prediction."""
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute the F1 score."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    return compute_f1_from_tokens(gold_toks, pred_toks)
