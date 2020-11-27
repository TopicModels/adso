"""Vectorizer class."""

from __future__ import annotations

from typing import List, Union

import nltk

from ..data import Dataset
from . import nltk_download
from .common import Transformer


def _lowercase(s: str) -> str:
    return s.lower()


class Tokenizer(Transformer):
    def __init__(
        self: Tokenizer,
        tokenizer: callable = nltk.tokenize.word_tokenize,
        stemmer: Union[None, callable] = _lowercase,
        stopwords: Union[None, List[str]] = None,
    ) -> None:

        if tokenizer == nltk.tokenize.word_tokenize:
            nltk_download("punkt")
        self.tokenizer = tokenizer

        self.stemmer = stemmer

        self.stopwords = stopwords

        if self.stemmer and self.stopwords:
            self.transformer = lambda s: list(
                filter(
                    lambda tkn: tkn not in self.stopwords,
                    map(self.stemmer, self.tokenizer(s)),
                )
            )
        elif self.stopwords:
            self.transformer = lambda s: list(
                filter(lambda tkn: tkn not in self.stopwords, self.tokenizer(s))
            )
        elif self.stemmer:
            self.transformer = lambda s: list(map(self.stemmer, self.tokenizer(s)))
        else:
            self.transformer = self.tokenizer

    def fit(self: Tokenizer, data: Dataset) -> None:
        pass

    def transform(self: Tokenizer, data: Dataset):
        return list(map(self.transformer, data.get_data()))

    def fit_transform(self: Tokenizer, data: Dataset):
        return self.transform(data)
