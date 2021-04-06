"""Vectorizer class."""

from __future__ import annotations

from typing import Callable, List, Union

import nltk

from . import nltk_download
from .common import Transformer
from ..data import Dataset


def _lowercase(s: str) -> str:
    return s.lower()


class Tokenizer(Transformer):
    """Transform each document in a corpus in a list of tokens."""

    def __init__(
        self: Tokenizer,
        tokenizer: Callable[[str], List[str]] = nltk.tokenize.word_tokenize,
        stemmer: Union[None, Callable[[str], str]] = _lowercase,
        stopwords: Union[None, List[str]] = None,
    ) -> None:
        """Initialize the Tokenizer, specifing callable and parameters to be used.

        Args:
            tokenizer (callable, optional): any function which maps a string into
                a list of strings. Defaults to :func:`nltk.tokenize.word_tokenize`.
            stemmer (Union[None, callable], optional): any function which maps a string
                into a string, for example a stemmer. Applied after the tokenizer.
                Defaults to _lowercase.
            stopwords (Union[None, List[str]], optional): list of string to be ignored.
                Removed after calling the stemmer. Defaults to None.
        """
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
        """Do nothing method."""
        pass

    def transform(self: Tokenizer, data: Dataset) -> List[List[str]]:
        """Tokenize a given Dataset.

        Args:
            data (Dataset): a :class:`data.Dataset` to be tokenized

        Returns:
            List[List[str]]: a list of list of tokens. Outer list map to documents
                while inner list map to words in each document.
        """
        return list(map(self.transformer, data.get_data()))

    def fit_transform(self: Tokenizer, data: Dataset) -> List[List[str]]:
        """Alias for :func:`~transform.Tokenizer.transform`.

        Args:
            data (Dataset): a :class:`data.Dataset` to be tokenized

        Returns:
            List[List[str]]: a list of list of tokens. Outer list map to documents
                while inner list map to words in each document.
        """
        return self.transform(data)
