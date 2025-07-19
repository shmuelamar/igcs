import gzip
import json
import logging
import math
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Iterable

import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm

from igcs import configs
from igcs.configs import (
    ENRON_FNAME,
    GITHUB_FNAME,
    MNEWS_FNAME,
    PG19_FNAME,
    PUBMED_FNAME,
    SAMPLED_RI_RAW_DATA_FNAME,
    TRIPADVISOR_FNAME,
    WIKI_FNAME,
)
from igcs.entities import Doc
from igcs.utils import ioutils, log
from igcs.utils.diskcache import disk_cache

SEED = 42

logger = logging.getLogger(__name__)


@disk_cache(cache_dir=configs.CACHE_DIR / "word_counts")
def get_word_count(docs: tuple[tuple[str, ...] | str, ...]) -> tuple[int, ...]:
    word_counts = []
    for x in tqdm(docs):
        if isinstance(x, str):
            x = (x,)

        word_counts.append(sum(len(word_tokenize(xi)) for xi in x))
    return tuple(word_counts)


def sample_pg19_books(n: int, min_words: int, max_words: int, fname: Path = PG19_FNAME):
    logger.info(f"sampling {n} books samples")
    df = pd.read_json(fname, lines=True)
    df = df[(min_words <= df["word_count"]) & (df["word_count"] <= max_words)]
    for row in df.sample(n, random_state=SEED).itertuples(index=False):
        yield {
            "type": "book",
            "id": f"pg-19.{row.id}",
            "documents": [Doc(id=0, text=row.content)],
        }


def sample_enron_emails(n: int, min_words: int, max_words: int, fname: Path = ENRON_FNAME):
    logger.info(f"sampling {n} email threads samples")
    emails_df = pd.read_json(fname, lines=True)
    emails_df["has_chain"] = emails_df["chain_count"].apply(bool)
    emails_df = emails_df[
        (min_words <= emails_df["word_count"]) & (emails_df["word_count"] <= max_words)
    ]

    samples_with_chain = emails_df[emails_df["has_chain"]].sample(
        math.ceil(n / 2), random_state=SEED
    )
    samples_wo_chain = emails_df[~emails_df["has_chain"]].sample(
        math.floor(n / 2), random_state=SEED
    )
    all_samples = (
        pd.concat([samples_wo_chain, samples_with_chain])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    for row in all_samples.itertuples(index=False):
        row_id = "/".join(Path(row.id).parts[-3:])
        yield {
            "type": "email",
            "id": f"enron.{row_id}",
            "documents": [Doc(id=0, text=row.content)],
        }


def sample_multinews_articles(n: int, min_words: int, max_words: int, fname: Path = MNEWS_FNAME):
    logger.info(f"sampling {n} news articles samples")
    df = pd.read_json(fname, lines=True)

    raw_docs = tuple(tuple("\n".join(d["text"]) for d in docs) for docs in df["documents"].tolist())
    df["word_count"] = get_word_count(docs=raw_docs)

    df = df[(min_words <= df["word_count"]) & (df["word_count"] <= max_words)]
    for row in df.sample(n, random_state=SEED).itertuples(index=False):
        row_id = f"multinews.{row.instance_id}"
        yield {
            "type": "news_article",
            "id": f"multinews.{row_id}",
            "documents": [
                Doc(id=doc_id, text="\n".join(doc["text"]))
                for doc_id, doc in enumerate(row.documents)
            ],
        }


def sample_tripadvisor_reviews(
    n: int, min_words: int, max_words: int, fname: Path = TRIPADVISOR_FNAME
):
    logger.info(f"sampling {n} reviews samples")
    df = pd.read_json(fname, lines=True)

    raw_docs = tuple(tuple(d["Content"] for d in docs) for docs in df["Reviews"].tolist())
    df["word_count"] = get_word_count(docs=raw_docs)
    df = df[(min_words <= df["word_count"]) & (df["word_count"] <= max_words)]

    for row in df.sample(n, random_state=SEED).itertuples(index=False):
        documents = [
            Doc(
                id=i,
                text=review["Content"],
                filename=review.get("Title", "").strip("“").strip("”"),
                metadata={"review_id": review["ReviewID"]},
            )
            for i, review in enumerate(row.Reviews)
        ]
        yield {
            "type": "hotel_reviews",
            "id": f'tripadvisor.{row.HotelInfo["HotelID"]}',
            "documents": documents,
        }


def sample_github_code(n: int, min_words: int, max_words: int, fname: Path = GITHUB_FNAME):
    logger.info(f"sampling {n} source-code files samples")
    df = pd.read_json(fname, lines=True)

    raw_docs = tuple(tuple(d["content"] for d in docs) for docs in df["documents"].tolist())
    df["word_count"] = get_word_count(docs=raw_docs)
    df = df[(min_words <= df["word_count"]) & (df["word_count"] <= max_words)]

    for row in df.sample(n, random_state=SEED).itertuples(index=False):
        documents = [
            Doc(
                id=i,
                text=code["content"],
                filename=code["filename"],
                metadata={"file_id": row.repo_name + "/" + code["filename"]},
            )
            for i, code in enumerate(row.documents)
        ]
        yield {
            "type": "code",
            "id": f"github.{row.cluster_id}",
            "documents": documents,
        }


def sample_wiki_pages(
    n: int,
    min_words: int,
    max_words: int,
    fname: Path = WIKI_FNAME,
    max_paragraphs: int = 30,
):
    logger.info(f"sampling {n} wiki pages samples")

    df = pd.read_json(fname, lines=True)

    df["paragraphs"] = df.apply(
        lambda row: [row["title"]] + [p for p in row["text"].split("\n\n")], axis=1
    )
    df["content"] = df["paragraphs"].apply(lambda ps: "\n\n".join(ps[:max_paragraphs]).strip())

    raw_docs = tuple(df["content"].tolist())
    df["word_count"] = get_word_count(docs=raw_docs)

    df = df[(min_words <= df["word_count"]) & (df["word_count"] <= max_words)]
    for row in df.sample(n, random_state=SEED).itertuples(index=False):
        yield {
            "type": "wiki",
            "id": f"wiki.{row.id}",
            "documents": [Doc(id=0, text=row.content)],
        }


def sample_pubmed_abstract(n: int, min_words: int, max_words: int, fname: Path = PUBMED_FNAME):
    logger.info(f"sampling {n} pubmed abstracts samples")
    data = ioutils.jsonl_load(fname)

    df = pd.DataFrame(
        [
            {
                "content": x["MedlineCitation"]
                .get("Article", {})
                .get("Abstract", {})
                .get("AbstractText")
                or "",
                "title": x["MedlineCitation"].get("Article", {}).get("ArticleTitle") or "",
                "id": x["MedlineCitation"]["PMID"],
            }
            for x in data
        ]
    )
    assert df["id"].is_unique

    raw_docs = tuple(df["content"].tolist())
    df["word_count"] = get_word_count(docs=raw_docs)

    df = df[(min_words <= df["word_count"]) & (df["word_count"] <= max_words)]
    for row in df.sample(n, random_state=SEED).itertuples(index=False):
        yield {
            "type": "abstract",
            "id": f"pubmed.{row.id}",
            "documents": [Doc(id=0, text=f"{row.title}\n\n{row.content}")],
        }


def set_negative_examples(src_samples: Iterable[dict], n_neg: int) -> list[dict]:
    src_samples = list(src_samples)
    neg_samples_start_pos = 200
    neg_samples_end_pos = neg_samples_start_pos + n_neg

    for i, sample in enumerate(src_samples):
        sample["is_negative"] = bool(neg_samples_start_pos <= i < neg_samples_end_pos)
    return src_samples


def main(
    n_pos=475,
    n_neg=25,
    min_words=350,
    max_words=3500,
    outfile=SAMPLED_RI_RAW_DATA_FNAME,
):
    n = n_pos + n_neg

    # return nameless multi-document
    news = sample_multinews_articles(n, min_words, max_words)

    # return single, nameless doc content
    books = sample_pg19_books(n, min_words, max_words)
    emails = sample_enron_emails(n, min_words, max_words)
    wiki = sample_wiki_pages(n, min_words, max_words)
    abstracts = sample_pubmed_abstract(n, min_words, max_words)

    # return multi-document with title
    reviews = sample_tripadvisor_reviews(n, min_words, max_words)
    codes = sample_github_code(n, min_words, max_words)

    all_sources = [
        set_negative_examples(src, n_neg)
        for src in [books, emails, wiki, abstracts, news, reviews, codes]
    ]
    data = list(chain(*all_sources))
    logger.info(f"negative statistics: {Counter(x['is_negative'] for x in data)}")

    logger.info(f"writing {len(data)} samples into {outfile}")
    with gzip.open(outfile, "wt") as fp:
        for row in data:
            row["documents"] = [doc.dict() for doc in row["documents"]]
            fp.write(json.dumps(row))
            fp.write("\n")


if __name__ == "__main__":  # pragma: nocover
    log.init()
    main()
