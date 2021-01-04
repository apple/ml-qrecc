#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
This script downloads webpages in the conversation contexts from the Wayback Machine.
"""

from argparse import ArgumentParser
import glob
import json
import logging
import multiprocessing
from pathlib import Path
import random
import re
import requests
from requests.exceptions import HTTPError
import shutil
import time
import urllib.parse
import uuid

from bs4 import BeautifulSoup
import pandas as pd


wayback_prefix = re.compile(r'^https:\/\/web\.archive\.org\/web')
replace_pattern = re.compile(r'(web\.archive\.org\/web\/\d+)')
blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head',
    'input',
    'script',
    'style',
    # there may be more elements we don't want
]


def download_with_retry(url: str, max_retries: int = 10) -> requests.Response:
    """Download a URL with exponential backoff, until max_retries is reached."""
    retry_num = 0
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except HTTPError as e:
            status_code = e.response.status_code
            if not (status_code == 429 or status_code >= 500):
                # This is not an error we should retry on
                raise e

            if retry_num > max_retries:
                logging.error(
                    f'Failed to perform GET request on {url} after {max_retries} retries.'
                )
                raise e

            if status_code == 429:
                time.sleep(5 + 2 ** retry_num + random.randint(0, 1000) / 1000)
            else:
                time.sleep(2 ** retry_num + random.randint(0, 1000) / 1000)
            retry_num += 1


def extract_text(html_text: str) -> str:
    """Extracts text from an HTML document."""
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.find_all(text=True)
    output = ''
    for t in text:
        if t.parent.name not in blacklist:
            output += f'{t} '

    return output


def download_link(tup):
    link = tup[0]
    output_path = tup[1]
    num_workers = tup[2]
    page_id = str(uuid.uuid4())
    url_no_header = None

    try:
        # Find the Wayback Machine link
        if not wayback_prefix.match(link):
            link_encoded = urllib.parse.quote(link)

            available, availability_attempt = False, 0
            # Sometimes the API returns HTTP success code 200, but archived snapshots shows page is unavailable
            # when it actually is. Give it a total of three tries.
            while not available and availability_attempt < 3:
                response = download_with_retry(
                    f'http://archive.org/wayback/available?url={link_encoded}&timestamp=20191127'
                )
                json_response = response.json()
                available = 'closest' in json_response['archived_snapshots']
                availability_attempt += 1

            if not available:
                logging.warning(
                    f'Not available on Wayback Machine: {link}, HTTP code {response.status_code}, {json_response}'
                )
                return {'link': link, 'page_id': page_id, 'available': False}

            url = json_response['archived_snapshots']['closest']['url']
        else:
            url = link

        match = replace_pattern.search(url)
        assert match
        url_no_header = replace_pattern.sub(f'{match.group(1)}id_', url)

        response = download_with_retry(url_no_header)
        html_page = response.text
        parsed_text = extract_text(html_page)

        proc = multiprocessing.current_process()
        pid_mod = str(proc.pid % num_workers)

        (output_path / pid_mod).mkdir(parents=True, exist_ok=True)

        with open(output_path / pid_mod / page_id, 'w') as f:
            doc = {
                'id': url_no_header,
                'contents': parsed_text,
            }
            f.write(json.dumps(doc) + '\n')

        return {
            'link': link,
            'page_id': page_id,
            'available': True,
            'status_code': response.status_code,
            'wayback_url': url_no_header,
        }
    except HTTPError as http_err:
        logging.warning(f'HTTP error occurred: {http_err} for {link}')
        return {
            'link': link,
            'page_id': page_id,
            'available': False,
            'status_code': http_err.response.status_code if http_err.response else None,
            'wayback_url': url_no_header,
        }
    except UnicodeDecodeError as e:
        logging.warning(f'Unicode decode error occurred: {e} for {link}')
        return {
            'link': link,
            'page_id': page_id,
            'available': False,
            'status_code': response.status_code,
            'wayback_url': url_no_header,
        }
    except Exception as e:
        logging.warning(f'Exception occurred: {e} for {link}')
        return {
            'link': link,
            'page_id': page_id,
            'available': False,
            'status_code': None,
            'wayback_url': url_no_header,
        }


def crawl_wayback_machine(
    inputs_globbing_pattern: str, output_dir: str, num_workers: int
) -> None:
    links = set()
    for dataset in glob.glob(inputs_globbing_pattern):
        with open(dataset) as f:
            data = json.load(f)
            for conversation_turn in data:
                if conversation_turn['Answer_URL'] == '':
                    continue

                for url in conversation_turn['Answer_URL'].split(' '):
                    if url.endswith('.pdf'):
                        continue

                    anchor_sign_pos = url.find('#')
                    if anchor_sign_pos != -1:
                        url = url.split('#')[0]

                    links.add(url)
    links = list(links)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = []
    with multiprocessing.Pool(num_workers) as p:
        for i, result in enumerate(
            p.imap_unordered(
                download_link,
                [(l, output_path, num_workers) for l in links],
                chunksize=16,
            )
        ):
            records.append(result)
            if (i + 1) % 10 == 0:
                logging.info(f'Processed {i + 1} / {len(links)} links...')

    # Combine small files together into larger files
    for worker_output_dir in output_path.iterdir():
        if worker_output_dir.is_dir():
            with open(output_path / f'{worker_output_dir.name}.jsonl', 'w') as outfile:
                for single_doc_file in worker_output_dir.iterdir():
                    with open(single_doc_file) as infile:
                        outfile.write(infile.read())

            shutil.rmtree(worker_output_dir)

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_path / 'summary.tsv', index=False, sep='\t')


if __name__ == '__main__':
    parser = ArgumentParser(description='Crawl pages from Wayback Machine')
    parser.add_argument(
        '--inputs', required=True, help='Globbing pattern for train and test JSON files'
    )
    parser.add_argument(
        '--output-directory',
        required=True,
        help='Path to directory containing crawled output',
    )
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='Number of workers for downloading in parallel',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    crawl_wayback_machine(args.inputs, args.output_directory, args.workers)
