#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
This script creates a corpus of documents from the November 2019 Common Crawl archive.
"""

from argparse import ArgumentParser
from collections import defaultdict
import gzip
import json
import logging
from multiprocessing import Pool
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Dict, List, Tuple
import urllib.request

from warcio.archiveiterator import ArchiveIterator

index_files_root = Path('index-files')
filter_lists_root = Path('filter-lists')
sampled_filter_lists_root = Path('filter-lists-sampled')
wet_files_cache = Path('wet-files')


def get_cc_index_paths() -> List[str]:
    """Get a list of paths for Common Crawl URL index files."""
    index_paths = []
    with tempfile.NamedTemporaryFile() as temp_f:
        urllib.request.urlretrieve(
            'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2019-47/cc-index.paths.gz',
            temp_f.name,
        )
        with gzip.open(temp_f.name, 'rb') as f:
            for line in f:
                line = line.decode('utf-8').rstrip()
                if line.endswith('.gz'):
                    index_paths.append(f'https://commoncrawl.s3.amazonaws.com/{line}')

    return index_paths


def get_cc_wet_paths() -> Dict[str, str]:
    """Get a dict of WET file name to WET URL."""
    wet_urls = {}
    with tempfile.NamedTemporaryFile() as temp_f:
        urllib.request.urlretrieve(
            'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2019-47/wet.paths.gz',
            temp_f.name,
        )
        with gzip.open(temp_f.name, 'rb') as f:
            for line in f:
                line = line.decode('utf-8').rstrip()
                filename = line.split('/')[-1]
                wet_urls[filename] = f'https://commoncrawl.s3.amazonaws.com/{line}'

    return wet_urls


def process_cc_index(index_url: str) -> Dict[str, List[str]]:
    """Return a map of WET file to list of URLs it contains."""
    # Download index file
    filename = index_url.split('/')[-1]
    index_files_root.mkdir(exist_ok=True)
    if not (index_files_root / filename).exists():
        urllib.request.urlretrieve(index_url, index_files_root / filename)

    # Parse index file
    wet_to_urls = defaultdict(list)
    cc_index_line_pattern = re.compile(r'^[\S]+ \d+ (.*)$')
    with gzip.open(index_files_root / filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8').rstrip()
            match = cc_index_line_pattern.match(line)
            if match:
                url_metadata = json.loads(match.group(1))
                if (
                    url_metadata['status'] == '200'
                    and url_metadata.get('languages') == 'eng'
                    and url_metadata['mime'] == 'text/html'
                ):
                    wet_filename = url_metadata['filename'].split('/')[-1]
                    wet_to_urls[wet_filename].append(url_metadata['url'])
            else:
                logging.error(f'Line in index file cannot be matched by regex: {line}')

    return wet_to_urls


def sort_and_sample_filter_list(filter_list_path: Path) -> None:
    """Sort and sample URLs in a filter list."""
    urls = []
    with open(filter_list_path) as f:
        for line in f:
            urls.append(line.rstrip())

    urls.sort()

    with open(sampled_filter_lists_root / filter_list_path.name, 'w') as f:
        for i, url in enumerate(urls):
            if i % 100 == 0:
                f.write(url + '\n')


def sample_filter_lists() -> None:
    """Sample filter lists."""
    filter_lists = list(filter_lists_root.iterdir())
    sampled_filter_lists_root.mkdir(exist_ok=True)

    with Pool() as p:
        p.map(sort_and_sample_filter_list, filter_lists)


def process_wet_file(tup: Tuple[Path, str, str, Path],) -> None:
    """Download WET file and extract webpages from WARC whose URL is in the filter list."""
    filter_list, wet_name, wet_url, commoncrawl_docs_root = tup
    accepted_urls = set()
    with open(filter_list) as f:
        for line in f:
            accepted_urls.add(line.rstrip())

    attempt = 0
    while attempt < 3:
        try:
            urllib.request.urlretrieve(wet_url, wet_files_cache / wet_name)
            break
        except Exception:
            logging.exception(f'Error while downloading {wet_url}')
            attempt += 1

    if not (wet_files_cache / wet_name).exists():
        logging.error(
            f'Failed to download {wet_url} after 3 attempts. Ignoring file...'
        )
        return

    with gzip.open(wet_files_cache / wet_name, 'rb') as stream, open(
        commoncrawl_docs_root / f'{wet_name}.jsonl', 'w'
    ) as f:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'conversion':
                url = record.rec_headers.get_header('WARC-Target-URI')
                if url not in accepted_urls:
                    continue

                contents = record.content_stream().read().decode('utf-8')
                if contents.startswith('404 Not Found'):
                    continue

                output_dict = {'id': url, 'contents': contents}

                f.write(json.dumps(output_dict) + '\n')

    os.remove(wet_files_cache / wet_name)


def get_docs_from_wet_files(parallelism, commoncrawl_docs_root: Path) -> None:
    """Download WET files and extract webpages whose URLs is in the filter list."""
    wet_files_cache.mkdir(exist_ok=True)
    commoncrawl_docs_root.mkdir(exist_ok=True, parents=True)

    filter_lists = list(sampled_filter_lists_root.iterdir())

    # Download WET file paths
    wet_paths = get_cc_wet_paths()
    wet_names = []
    resolved_wet_paths = []
    for filter_list in filter_lists:
        wet_filename = str(filter_list.name).replace('.warc.gz.txt', '.warc.wet.gz')
        wet_names.append(wet_filename)
        resolved_wet_paths.append(wet_paths[wet_filename])

    with Pool(parallelism) as p:
        for i, _ in enumerate(
            p.imap_unordered(
                process_wet_file,
                zip(
                    filter_lists,
                    wet_names,
                    resolved_wet_paths,
                    [commoncrawl_docs_root for _ in range(len(filter_lists))],
                ),
            )
        ):
            if (i + 1) % 50 == 0:
                logging.info(f'Processed {i + 1} / {len(filter_lists)} WET files...')


def main(parallelism: int, commoncrawl_docs_root: Path):
    cc_index_paths = get_cc_index_paths()

    # Construct filter lists
    if filter_lists_root.exists():
        shutil.rmtree(filter_lists_root)
    filter_lists_root.mkdir(exist_ok=True)

    for i in range(0, len(cc_index_paths), parallelism):
        with Pool(parallelism) as p:
            logging.info(
                f'Processing Common Crawl index {i+1}-{min(i + parallelism, len(cc_index_paths))} / {len(cc_index_paths)}...'
            )
            partial_filter_lists = p.map(
                process_cc_index, cc_index_paths[i : i + parallelism]
            )
            for partial_filter_list in partial_filter_lists:
                for wet_filename, urls in partial_filter_list.items():
                    with open(filter_lists_root / f'{wet_filename}.txt', 'a') as f:
                        for url in urls:
                            f.writelines(url + '\n')

    # Create sampled filter lists
    logging.info('Sorting and sampling filter lists...')
    sample_filter_lists()

    # Download WET files and filter records
    logging.info('Processing WET files...')
    get_docs_from_wet_files(parallelism, commoncrawl_docs_root)

    # Remove temporary files
    logging.info('Done processing WET files, removing temporary directories...')
    shutil.rmtree(index_files_root)
    shutil.rmtree(filter_lists_root)
    shutil.rmtree(sampled_filter_lists_root)
    shutil.rmtree(wet_files_cache)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Creates a corpus of documents from the November 2019 Common Crawl archive'
    )
    parser.add_argument(
        '--output-directory',
        default='docs/common-crawl',
        help='Path to directory containing document output, defaults to docs/common-crawl',
    )
    parser.add_argument(
        '--workers',
        default=8,
        type=int,
        help='Number of workers for downloading in parallel',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    commoncrawl_docs_root = Path(args.output_directory)

    main(args.workers, commoncrawl_docs_root)
