"""Vectorizer class."""

from __future__ import annotations
from typing import Union

import nltk

from ..data import Dataset
from . import nltk_download


def _lowercase(s: str) -> str:
    return s.lower()


class Tokenizer:
    def __init__(
        self: Tokenizer,
        tokenizer: callable = nltk.tokenize.word_tokenize,
        stemmer: Union[None, callable] = _lowercase,
    ) -> None:

        if tokenizer == nltk.tokenize.word_tokenize:
            nltk_download("punkt")
        self.tokenizer = tokenizer

        self.stemmer = stemmer

        if self.stemmer:
            self.transformer = lambda s: list(map(self.stemmer, self.tokenizer(s)))
        else:
            self.transformer = self.tokenizer

    def fit(self: Tokenizer, data: Dataset) -> None:
        pass

    def transform(self: Tokenizer, data: Dataset):
        return list(map(self.transformer, data.get_data()))

    def fit_transform(self: Tokenizer, data: Dataset):
        return self.transform(data)
