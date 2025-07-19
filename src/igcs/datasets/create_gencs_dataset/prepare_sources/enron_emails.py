import email
import logging
import os
from glob import glob

import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm

from igcs import configs

logger = logging.getLogger("igcs.create_gencs_dataset.enron_emails")


def load_enron_emails_as_df(enron_input_dir: str) -> pd.DataFrame:
    fnames = list(glob(os.path.join(enron_input_dir, "**", "*"), recursive=True))

    logger.info(f"found {len(fnames)} email files")
    messages = []
    for fname in tqdm(fnames):
        if not os.path.isfile(fname):
            continue

        file = "/".join(os.path.split(fname)[-3:])
        with open(fname, "rb") as fp:
            msg = email.message_from_binary_file(fp)

        messages.append({"file": file, "message": msg})

    return pd.DataFrame(messages)


## Helper functions
# Adapted from: https://www.kaggle.com/code/abhishekpattanayak/the-enron-corpus-dataset
def get_text_from_email(msg):
    """To get the content from email objects"""
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "".join(parts)


def split_email_addresses(line):
    """To separate multiple email addresses"""
    line = str(line).strip()
    if line:
        addrs = line.split(",")
        addrs = list(frozenset(map(str.strip, addrs)))
    else:
        addrs = None
    return addrs


def chain_count(content: str) -> int:
    return content.count("-----Original Message-----")


def parse_and_load_enron_emails(enron_input_dir: str) -> pd.DataFrame:
    raw_df = load_enron_emails_as_df(enron_input_dir)

    logger.info(f"parse content from emails")
    raw_df["content"] = raw_df["message"].apply(get_text_from_email)

    logger.info("parse email headers")
    df = pd.json_normalize([dict(x) for x in tqdm(raw_df["message"])], max_level=1)

    logger.info("Split multiple email addresses")
    df["From"] = df["From"].map(split_email_addresses).apply(lambda x: list(x)[0])
    df["To"] = df["To"].map(split_email_addresses)

    logger.info("get user and content from raw data")
    df["user"] = raw_df["file"].map(lambda x: x.split("/")[0])
    df["content"] = raw_df["content"].astype("string")
    df["id"] = raw_df["file"]

    logger.info("removing technical fields")
    df.drop(
        ["Message-ID", "Mime-Version", "Content-Transfer-Encoding"],
        inplace=True,
        axis=1,
    )

    # add some stats
    logger.info("counting email chains")
    df["chain_count"] = df["content"].apply(chain_count)

    logger.info("counting email words")
    df["word_count"] = [len(word_tokenize(x)) for x in tqdm(df["content"])]

    return df


def main(
    indir=configs.RAW_TRAIN_DATASETS_DIR / "enron-emails" / "maildir",
    outfile=configs.RAW_TRAIN_DATASETS_DIR / "enron_emails.jsonl.gz",
):
    # parse Enron emails into a file later to be sampled as required
    df = parse_and_load_enron_emails(indir)
    logger.info(f"saving to {outfile}")
    df.to_json(outfile, lines=True, compression="gzip", orient="records")


if __name__ == "__main__":  # pragma: nocover
    main()
