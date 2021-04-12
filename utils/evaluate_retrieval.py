#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

from argparse import ArgumentParser
import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from evaluate_qa import compute_exact, compute_f1
from span_heuristic import find_closest_span_match

"""
Functions for evaluating passage retrieval.

This is used to compute MRR (mean reciprocal rank), Recall@10, and Recall@100 in Table 5 of the paper.
"""


RELEVANCE_THRESHOLD = 0.8


def compute_f1_for_retrieved_passage(line: str) -> dict:
    """
    Given a serialized JSON line, with fields 'content' and 'answer', find the closest span matching answer,
    update the deserialized dict with the span and F1 score, and return the dict.
    """
    data = json.loads(line)
    content, answer = data['content'], data['answer']

    # If there is no answer, although the closest extractive answer is '', in the MRR and recall@k functions below
    # we do not count any passage for these questions as relevant.
    if len(answer) < 1:
        data['heuristic_answer'] = ''
        data['f1'] = compute_f1(answer, '')
        return data

    best_span, best_f1 = find_closest_span_match(content, answer)

    data['heuristic_answer'] = best_span
    data['f1'] = best_f1

    return data


def compute_mean_reciprocal_rank(
    question_id_to_docs: Dict[str, List[dict]], relevance_threshold: float
) -> float:
    """Given a dictionary mapping a question id to a list of docs, find the mean reciprocal rank."""
    recip_rank_sum = 0
    for qid, docs in question_id_to_docs.items():
        top_rank = float('inf')
        for doc in docs:
            if len(doc['answer']) > 0 and doc['f1'] >= relevance_threshold:
                top_rank = min(top_rank, doc['rank'])

        recip_rank = 1 / top_rank if top_rank != float('inf') else 0
        recip_rank_sum += recip_rank

    return recip_rank_sum / len(question_id_to_docs)


def compute_recall_at_k(
    question_id_to_docs: Dict[str, List[dict]], k: int, relevance_threshold: float
) -> float:
    """
    Given a dictionary mapping a question id to a list of docs, find the recall@k.

    We define recall@k = 1.0 if any document in the top-k is relevant, and 0 otherwise.
    """
    relevant_doc_found_total = 0
    for qid, docs in question_id_to_docs.items():
        relevant_doc_found = 0
        for doc in docs:
            if len(doc['answer']) > 0 and doc['f1'] >= relevance_threshold and doc['rank'] <= k:
                relevant_doc_found = 1
                break

        relevant_doc_found_total += relevant_doc_found

    return relevant_doc_found_total / len(question_id_to_docs)


def compute_extractive_upper_bounds(
    question_id_to_docs: Dict[str, List[dict]], temp_files_directory: Path
) -> Tuple[float, float]:
    """Given a dictionary mapping a question id to a list of docs, find the extractive upper bounds of (EM, F1)."""
    total_em, total_f1 = 0, 0.0
    with open(temp_files_directory / 'retrieved-passages-relevant-f1.jsonl', 'w') as outfile:
        for qid, docs in question_id_to_docs.items():
            best_em, best_f1 = 0, 0.0
            best_doc = docs[0]
            for doc in docs:
                em = compute_exact(doc['answer'], doc['heuristic_answer'])
                f1 = compute_f1(doc['answer'], doc['heuristic_answer'])
                if f1 > best_f1:
                    best_doc = doc
                best_em = max(best_em, em)
                best_f1 = max(best_f1, f1)
                if best_em == 1 and best_f1 == 1.0:
                    break

            total_em += best_em
            total_f1 += best_f1

            outfile.write(json.dumps(best_doc) + '\n')

    return (
        total_em / len(question_id_to_docs),
        total_f1 / len(question_id_to_docs),
    )


def get_unique_relevant_docs_count(
    question_id_to_docs: Dict[str, List[dict]], relevance_threshold: float
) -> float:
    """Given a dictionary mapping a question id to a list of docs, find the number of unique relevant docs."""
    unique_relevant_docs = set()
    for qid, docs in question_id_to_docs.items():
        for doc in docs:
            if len(doc['answer']) > 0 and doc['f1'] >= relevance_threshold:
                unique_relevant_docs.add(doc['docid'])

    return len(unique_relevant_docs)


def get_average_relevant_docs_per_question(
    question_id_to_docs: Dict[str, List[dict]], relevance_threshold: float
) -> float:
    """Given a dictionary mapping a question id to a list of docs, find the average number of relevant docs per question."""
    relevant_docs = 0
    for qid, docs in question_id_to_docs.items():
        for doc in docs:
            if len(doc['answer']) > 0 and doc['f1'] >= relevance_threshold:
                relevant_docs += 1

    return relevant_docs / len(question_id_to_docs)


def main(retrieved_passages_pattern: str, temp_files_directory: str, workers: int):
    retrieved_passages_files = Path().glob(retrieved_passages_pattern)
    temp_files_directory = Path(temp_files_directory)
    temp_files_directory.mkdir(exist_ok=True, parents=True)

    question_id_to_docs = {}

    for retrieved_passages_file in retrieved_passages_files:
        with open(retrieved_passages_file) as infile:
            with Pool(workers) as p:
                for i, passage_results in enumerate(
                    p.imap(compute_f1_for_retrieved_passage, infile)
                ):
                    if (i + 1) % 5000 == 0:
                        logging.info(
                            f'Processing {retrieved_passages_file.name}, {i + 1} lines done...'
                        )

                    qid = f"{passage_results['Conversation-ID']}_{passage_results['Turn-ID']}"
                    if qid not in question_id_to_docs:
                        question_id_to_docs[qid] = []

                    question_id_to_docs[qid].append(
                        {
                            'Conversation-ID': passage_results['Conversation-ID'],
                            'Turn-ID': passage_results['Turn-ID'],
                            'docid': passage_results['docid'],
                            'content': passage_results['content'],
                            'rank': passage_results['rank'],
                            'answer': passage_results['answer'],
                            'heuristic_answer': passage_results['heuristic_answer'],
                            'f1': passage_results['f1'],
                        }
                    )

    print('Final metrics:')
    unique_relevant_docs = get_unique_relevant_docs_count(question_id_to_docs, RELEVANCE_THRESHOLD)
    unique_docs_perfect_f1 = get_unique_relevant_docs_count(question_id_to_docs, 1.0)
    avg_relevant_docs_per_question = get_average_relevant_docs_per_question(
        question_id_to_docs, 1.0
    )

    print(f'Total number of unique queries: {len(question_id_to_docs)}')
    print(f'Total number of unique relevant docs: {unique_relevant_docs}')
    print(f'Total number of unique docs with F1=1.0: {unique_docs_perfect_f1}')
    print(f'Average number of relevant docs per query: {avg_relevant_docs_per_question}')

    mrr = compute_mean_reciprocal_rank(question_id_to_docs, RELEVANCE_THRESHOLD)
    recall_at_10 = compute_recall_at_k(question_id_to_docs, 10, RELEVANCE_THRESHOLD)
    recall_at_100 = compute_recall_at_k(question_id_to_docs, 100, RELEVANCE_THRESHOLD)
    print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')
    print(f'Recall@10: {recall_at_10 * 100:.2f}%')
    print(f'Recall@100: {recall_at_100 * 100:.2f}%')

    em_upper_bound, f1_upper_bound = compute_extractive_upper_bounds(
        question_id_to_docs, temp_files_directory
    )
    print(f'Extractive Upper Bound for EM (100 point scale): {em_upper_bound * 100:.2f}')
    print(f'Extractive Upper Bound for F1 (100 point scale): {f1_upper_bound * 100:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Passage retrieval evaluation')
    parser.add_argument(
        '--retrieved-passages-pattern',
        required=True,
        help="""A globbing pattern to select .jsonl files containing retrieved passages.
        Each json line should contain the fields 'Conversation-ID', 'Turn-ID', 'docid', 'content', 'answer', 'rank'.
        'answer' is the gold answer given in the QReCC dataset and rank is the rank of the document starting from 1.""",
    )
    parser.add_argument(
        '--temp-files-directory',
        default='/tmp/qrecc-retrieval-eval',
        help='Directory to store temporary files containing F1 scores, which can be used for debugging and analysis',
    )
    parser.add_argument(
        '--workers', default=8, type=int, help='Number of workers for parallel processing',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args.retrieved_passages_pattern, args.temp_files_directory, args.workers)
