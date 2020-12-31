# Building the Collection

This directory contains the scripts and instructions for downloading, processing, and building the collection for our baseline.

The collection consists of webpages from the Common Crawl and the Wayback Machine.
**Please note that the Common Crawl collection is [quite large](https://commoncrawl.org/2019/11/november-2019-crawl-archive-now-available/) (order of tens of TBs), so please check the data cap of your internet plan to make sure you stay within the limit.**

To download pages from the Common Crawl, run the following command.
This saves webpage documents in [JSON lines](https://jsonlines.org) (.jsonl) format into the `collection/commoncrawl` subdirectory.
For us this took slightly over a day to run.

```bash
time python download_commoncrawl_passages.py --output-directory collection/commoncrawl --workers 8
```

To download pages from the Wayback Machine, run the following command after you've extracted the dataset.
This saves webpage documents in .jsonl format into the `collection/wayback` subdirectory.
For us this took 9 hours to run.

```bash
time python download_wayback_passages.py --inputs '../dataset/*.json' --output-directory collection/wayback --workers 4
```

Next we segmented the webpage documents into smaller passages.
This is quite quick and took several minutes.

```bash
time python paragraph_chunker.py --input-directory collection --output-directory collection-paragraph --workers 8
```

Finally we indexed the passages using [Pyserini](https://github.com/castorini/pyserini/), a Python wrapper around [Anserini](http://anserini.io/), an information retrieval toolkit built on Lucene.
Java (JDK) is needed as a pre-requisite. 
After installing Pyserini we used the following command to build the index.
For us this took less than 2 hours.

```bash
time python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 76 -input collection-paragraph \
 -index index-paragraph -storePositions -storeDocvectors -storeRaw
```
