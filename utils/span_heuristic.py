#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
Heuristic for finding a span in some passage that's close to the golden span.
"""

from difflib import SequenceMatcher as SM
import re
import string
from typing import List, Tuple

from nltk.util import ngrams

from evaluate_qa import compute_f1, compute_f1_from_tokens, get_tokens, normalize_answer


ARTICLES_RE = re.compile(r'\b(a|an|the)\b', re.UNICODE)
EXCLUDED_PUNCTS = set(string.punctuation)


def _find_approximate_matching_sequence(context: str, target: str) -> Tuple[str, float]:
    """Find some substring in the context which closely matches the target, returning this substring with a score."""
    if target in context:
        return target, 1.0

    target_length = len(target.split())
    max_sim_val = 0
    max_sim_string = ''
    seq_matcher = SM()
    seq_matcher.set_seq2(target)
    for ngram in ngrams(context.split(), target_length + int(0.05 * target_length)):
        candidate_ngram = ' '.join(ngram)
        seq_matcher.set_seq1(candidate_ngram)
        similarity = seq_matcher.quick_ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = candidate_ngram
        if similarity == 1.0:
            # early exiting
            break

    return max_sim_string, max_sim_val


def _normalize_tokens(tokens: List[str], keep_empty_str=True) -> List[str]:
    """
    Normalize individual tokens.

    If keep_empty_str is True, this keeps the overall number of tokens the same.
    A particular token could be normalized to an empty string.
    """
    normalized_tokens = []
    for token in tokens:
        token = token.lower()
        token = ''.join(ch for ch in token if ch not in EXCLUDED_PUNCTS)
        token = re.sub(ARTICLES_RE, '', token)
        if keep_empty_str or len(token):
            normalized_tokens.append(token)

    return normalized_tokens


def find_closest_span_match(passage: str, gold_answer: str) -> Tuple[str, float]:
    """Heuristic for finding the closest span in a passage relative to some golden answer based on F1 score."""
    closest_encompassing_span, closest_encompassing_span_score = _find_approximate_matching_sequence(passage, gold_answer)
    closest_encompassing_span_tok = closest_encompassing_span.split()
    gold_answer_tok = gold_answer.split()
    closest_encompassing_span_tok_normalized = _normalize_tokens(closest_encompassing_span_tok)
    gold_answer_tok_normalized = _normalize_tokens(gold_answer_tok, keep_empty_str=False)

    best_span, best_score, best_i, best_j = '', 0, None, None
    for i in range(0, len(closest_encompassing_span_tok_normalized)):
        for j in range(i + 1, len(closest_encompassing_span_tok_normalized) + 1):
            score = compute_f1_from_tokens(
                gold_answer_tok_normalized,
                [t for t in closest_encompassing_span_tok_normalized[i:j] if len(t)],
            )
            if score > best_score:
                best_score = score
                best_i, best_j = i, j

    best_span = ' '.join(closest_encompassing_span_tok[best_i:best_j])
    best_f1 = compute_f1(gold_answer, best_span)
    return best_span, best_f1
