"""Text baseline utilities for caption analysis and TF-IDF experiments."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from projects.image_captioning.vocab import tokenize

DEFAULT_STOP_WORDS = {
    'a', 'an', 'the', 'in', 'on', 'is', 'are', 'and', 'of', 'to', 'with',
    'at', 'by', 'for', 'from', 'into', 'while', 'as', 'his', 'her', 'its',
    'their', 'this', 'that', 'there', 'be', 'being', 'has', 'have',
}


@dataclass(frozen=True)
class TfidfTerm:
    term: str
    tfidf: float
    frequency: int
    document_frequency: int


def compute_tfidf_terms(captions: list[str],
                        top_k: int = 30,
                        min_document_frequency: int = 2,
                        stop_words: set[str] | None = None) -> list[TfidfTerm]:
    """Compute corpus-level TF-IDF terms from captions without extra packages."""

    ignored_terms = DEFAULT_STOP_WORDS if stop_words is None else stop_words
    document_count = len(captions)
    term_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()

    for caption in captions:
        tokens = [token for token in tokenize(caption) if token not in ignored_terms]
        term_frequency.update(tokens)
        document_frequency.update(set(tokens))

    scored_terms: list[TfidfTerm] = []
    for term, frequency in term_frequency.items():
        df = document_frequency[term]
        if df < min_document_frequency:
            continue
        idf = math.log((1 + document_count) / (1 + df)) + 1
        tfidf = frequency * idf
        scored_terms.append(
            TfidfTerm(
                term=term,
                tfidf=tfidf,
                frequency=frequency,
                document_frequency=df,
            )
        )

    return sorted(
        scored_terms,
        key=lambda item: (item.tfidf, item.frequency),
        reverse=True,
    )[:top_k]
