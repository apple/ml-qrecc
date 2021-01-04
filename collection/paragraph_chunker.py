#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""For a directory of nested JSON lines files, where each line is a document, chunk each document into many passages."""

from argparse import ArgumentParser
import json
import logging
import multiprocessing
from pathlib import Path
from typing import List, Tuple

MIN_PASSAGE_TOKENS = 220


def chunk_doc(content: str) -> List[str]:
    """Given a document, return a list of passages of no fewer than MIN_PASSAGE_TOKENS tokens / passage until EOF."""
    passages = []
    passage_tokens = []
    lines = content.split('\n')
    for line in lines:
        line = line.rstrip()

        if '===' in line:
            continue
        if len(line) == 0:
            continue

        tokens = line.split()
        passage_tokens.extend(tokens)

        if len(passage_tokens) > MIN_PASSAGE_TOKENS:
            passages.append(' '.join(passage_tokens))
            passage_tokens = []

    passages.append(' '.join(passage_tokens))
    return passages


def process_file(tup: Tuple[str, str, Path]) -> None:
    """Chunk all documents in a single file."""
    input_directory, output_directory, input_file = tup
    output_file = str(input_file).replace(input_directory, output_directory)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file) as f1, open(output_path, 'w') as f2:
        for jsonl in f1:
            doc = json.loads(jsonl)
            passages = chunk_doc(doc['contents'])

            for i, passage in enumerate(passages):
                paragraph = {'id': f"{doc['id']}_p{i}", 'contents': passage}

                f2.write(json.dumps(paragraph) + '\n')


def chunk_documents(input_directory: str, output_directory: str, workers: int) -> None:
    """Iterate .jsonl files in input_directory and output .jsonl files in output_directory where each doc is chunked."""
    input_directory_path = Path(input_directory)

    jsonl_files = list(input_directory_path.glob('**/*.jsonl'))

    with multiprocessing.Pool(workers) as p:
        for i, _ in enumerate(
            p.imap_unordered(
                process_file,
                [(input_directory, output_directory, f) for f in jsonl_files],
                chunksize=16,
            )
        ):
            if (i + 1) % 100 == 0:
                logging.info(f'Processed {i + 1} / {len(jsonl_files)} files...')


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Chunk documents in .jsonl files into many passages.'
    )
    parser.add_argument(
        '--input-directory',
        required=True,
        help='Directory containing .jsonl files to chunk',
    )
    parser.add_argument(
        '--output-directory',
        required=True,
        help='Directory to store .jsonl files containing document passages',
    )
    parser.add_argument(
        '--workers',
        default=8,
        type=int,
        help='Number of workers for downloading in parallel',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    chunk_documents(args.input_directory, args.output_directory, args.workers)
