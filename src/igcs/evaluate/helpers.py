import re
import string
from functools import lru_cache
from typing import List

from nltk import word_tokenize

PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    text = s.lower()
    text = "".join(ch for ch in text if ch not in PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


@lru_cache(maxsize=int(2**20))
def normalized_unigram(s: str) -> List[str]:
    """returns normalized unigram tokens of the given string s"""
    return word_tokenize(normalize_answer(s))
