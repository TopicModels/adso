"""Vectorizer class."""

from __future__ import annotations

import abc
from collections import Counter
from functools import reduce
from itertools import chain, starmap
from typing import Dict, List, Union

import numpy as np

import scipy as sp

from .common import Transformer


class Vocab:  # Inspiraed by torchtext.vocab.Vocab
    """Vocabulary class, used to convert string to int and viceversa."""

    def __init__(
        self: Vocab,
        count: Counter,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        min_count: int = 1,
    ) -> None:
        """Construct the vocabulary from a :class:`collections.Counter` object with
        word count.

        Args:
            count (Counter): a :class:`collections.Counter` which store the list of
                words and their count, with which build the vocabulary.
            max_size (Union[None, int], optional): maximum number of word to store into
                the vocabulary. Most common words are kept.
                None to keep all words. Defaults to None.
            min_freq (float, optional): minimum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 0.
            max_freq (float, optional): maximum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 1.
            min_count (int, optional): minimum number of occurency (i.e. associated
                value in Counter object) to keep a word. Defaults to 1.

        Attributes:
            itos (Dict[int, str]): dictionary to access words in vocabulary
                given the index.
            stoi (Dict[str, int]): dictionary to access the index of a given
                word in vocabulary.
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_count = min_count

        total_count = sum(count.values())
        min_freq = max(self.min_freq * total_count, self.min_count)
        max_freq = self.min_freq * total_count

        words = dict(
            Counter(
                dict(
                    filter(
                        lambda x: (x[1] >= max_freq) or (x[1] <= min_freq),
                        count.items(),
                    )
                )
            ).most_common(max_size)
        ).keys()

        self.itos: Dict[int, str] = dict(enumerate(words))
        self.stoi: Dict[str, int] = dict(map(reversed, self.itos.items()))

    def __getitem__(self: Vocab, index: Union[int, str]) -> Union[str, int]:
        """Enable [ ] syntax for both integer (return correspondent word) and strings
        (returning correspondent index).

        Args:
            index (Union[int, str]): a word to get the correspondent inde or an index
                to get the correspondent word.

        Returns:
            Union[str, int]: the word or the index correspondent to the index or
                the word selected.
        """
        if isinstance(index, str):
            return self.stoi[index]
        elif isinstance(index, int):
            return self.itos[index]
        else:
            raise ValueError("index must be a string or an int")

    def __len__(self: Vocab) -> int:
        """Define len( ) for Vocab class.

        Returns:
            int: number of words in the vocabulary.
        """
        return len(self.stoi)


class Vectorizer(Transformer):
    """Abstract class to group vectorizer, i.e. text to number processors."""

    @abc.abstractmethod
    def fit(data: List[List[str]]) -> None:
        """Create the vocabulary.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.
        """
        pass

    @abc.abstractmethod
    def transform(data: List[List[str]]) -> Union[np.array, sp.sparse.spmatrix]:
        """Trasform the collection of tokens in a sequence of numerical values,
        given an already fitted vocabulary.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            Union[np.array, sp.sparse.spmatrix]: a matrix which maps documents to the
                rows.
        """
        pass

    @abc.abstractmethod
    def fit_transform(data: List[List[str]]) -> Union[np.array, sp.sparse.spmatrix]:
        """Trasform the collection of tokens in a sequence of numerical values,
        creating a vocabulary in the process.

         Args:
             data (List[List[str]]): a list of list of tokens. Outer list map to
                 documents while inner list map to words in each document. For example
                 the output of a :class:`transform.Tokenizer`.

         Returns:
             Union[np.array, sp.sparse.spmatrix]: a matrix which maps documents to the
                 rows.
        """
        pass


class CountVectorizer(Vectorizer):
    """Create a document-term matrix with count as values."""

    def __init__(
        self: CountVectorizer,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        min_count: int = 1,
    ) -> None:
        """Initialize the vectorizer and set the parameters.

        Args:
            max_size (Union[None, int], optional): maximum number of word to store into
                the vocabulary. Most common words are kept.
                None to keep all words. Defaults to None.
            min_freq (float, optional): minimum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 0.
            max_freq (float, optional): maximum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 1.
            min_count (int, optional): minimum number of occurency to keep a word.
                Defaults to 1.

        Attributes:
            vocab: the stored vocabulary used to transform the data.
        """
        if not ((0 <= min_freq <= 1) and (0 <= max_freq <= 1)):
            raise ValueError("min_freq and max_freq must be in the [0,1] interval")

        self.vocab: Vocab
        self.max_size = max_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_count = min_count

    def fit(
        self: CountVectorizer,
        data: List[List[str]],
    ) -> None:
        """Create the vocabulary.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.
        """
        count = reduce(lambda x, y: x + y, map(Counter, data))
        self.vocab = Vocab(
            count, self.max_size, self.min_freq, self.max_freq, self.min_count
        )

    def transform(self: CountVectorizer, data: List[List[str]]) -> sp.sparse.spmatrix:
        """Return a document-term matrix with count as values. Words not in vocabulary
        are discarded.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            sp.sparse.spmatrix: a document-term matrix with count as values.
        """
        if not self.vocab:
            NameError("It is necessary to fit before transform")

        def toindex(s: str) -> Union[None, str]:
            try:
                return self.vocab[s]
            except KeyError:
                return None

        def document_to_count(document: List[str]):
            return list(
                Counter(
                    list(
                        filter(
                            lambda tkn: tkn is not None,
                            map(toindex, document),
                        )
                    )
                ).items()
            )

        coo_sparse = np.array(
            list(
                chain.from_iterable(
                    starmap(
                        lambda row, doc: list(
                            starmap(lambda col, val: (val, row, col), doc)
                        ),
                        enumerate(map(document_to_count, data)),
                    )
                )
            )
        )

        return sp.sparse.csr_matrix(
            (coo_sparse[:, 0], (coo_sparse[:, 1], coo_sparse[:, 2])),
            shape=(len(data), len(self.vocab)),
            dtype=int,
        )

    def fit_transform(
        self: CountVectorizer,
        data: List[List[str]],
    ) -> Union[np.array, sp.sparse.spmatrix]:
        """Return a document-term matrix with count as values. Vocabulary is built
        on the data passed as input.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            sp.sparse.spmatrix: a document-term matrix with count as values.
        """
        self.fit(data)
        return self.transform(data)


class ListVectorizer(CountVectorizer):
    """Create a matrix with the ordered list of indeces for each document
    on the rows."""

    def __init__(
        self: ListVectorizer,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        min_count: int = 1,
        max_length: int = 100,
    ) -> None:
        """Initialize the vectorizer and set the parameters.

        Args:
            max_size (Union[None, int], optional): maximum number of word to store into
                the vocabulary. Most common words are kept.
                None to keep all words. Defaults to None.
            min_freq (float, optional): minimum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 0.
            max_freq (float, optional): maximum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 1.
            min_count (int, optional): minimum number of occurency to keep a word.
                Defaults to 1.
            max_length (int, optional): maximum lenght of each row, i.e. maximum number
                of words for each document to keep. The first N words are kept.
                Defaults to 100.

        Attributes:
            vocab: the stored vocabulary used to transform the data.
        """
        if not ((0 <= min_freq <= 1) and (0 <= max_freq <= 1)):
            raise ValueError("min_freq and max_freq must be in the [0,1] interval")

        super().__init__(
            max_size=max_size, min_freq=min_freq, max_freq=max_freq, min_count=min_count
        )
        self.max_length = max_length

    def fit(
        self: ListVectorizer,
        data: List[List[str]],
    ) -> None:
        """Create the vocabulary.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.
        """
        super().fit(data)

    def transform(self: ListVectorizer, data: List[List[str]]) -> np.array:
        """Return a matrix with the ordered list of indeces for each document
        on the rows. Words not in vocabulary are discarded.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            np.array: a matrix with the ordered list of indeces for each document
            on the rows.
        """
        if not self.vocab:
            NameError("It is necessary to fit before transform")

        def toindex(s: str) -> Union[None, str]:
            try:
                return self.vocab[s]
            except KeyError:
                return None

        def document_to_list(document: list(str)):
            return list(
                filter(
                    lambda tkn: tkn is not None,
                    map(toindex, document),
                )
            )

        data = list(map(document_to_list, data))
        length = min(max(map(len, data)), self.max_length)

        def pad(data: list, length: int):
            if len(data) > length:
                return data[:length]
            else:
                return data + [np.nan] * (length - len(data))

        return np.array(list(map(lambda row: pad(row, length), data)))

    def fit_transform(
        self: ListVectorizer,
        data: List[List[str]],
    ) -> np.array:
        """Return a matrix with the ordered list of indeces for each document
        on the rows. Vocabulary is built on the data passed as input.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            np.array: a matrix with the ordered list of indeces for each document
            on the rows.
        """
        self.fit(data)
        return self.transform(data)


class FreqVectorizer(CountVectorizer):
    """Create a document-term matrix with frequencies as values."""

    def __init__(
        self: FreqVectorizer,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        min_count: int = 1,
    ) -> None:
        """Initialize the vectorizer and set the parameters.

        Args:
            max_size (Union[None, int], optional): maximum number of word to store into
                the vocabulary. Most common words are kept.
                None to keep all words. Defaults to None.
            min_freq (float, optional): minimum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 0.
            max_freq (float, optional): maximum frequency (as count of word / sum of
                counts of all words) to keep a word. Defaults to 1.
            min_count (int, optional): minimum number of occurency to keep a word.
                Defaults to 1.

        Attributes:
            vocab: the stored vocabulary used to transform the data.
        """
        super().__init__(
            max_size=max_size, min_freq=min_freq, max_freq=max_freq, min_count=min_count
        )

    def fit(
        self: FreqVectorizer,
        data: List[List[str]],
    ) -> None:
        """Create the vocabulary.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.
        """
        super().fit(data)

    def transform(self: FreqVectorizer, data: List[List[str]]) -> sp.sparse.spmatrix:
        """Return a document-term matrix with frequencies as values. Words not in
        vocabulary are discarded.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            sp.sparse.spmatrix: a document-term matrix with frequencies as values.
        """
        matrix = super().transform(data)
        # row normalization
        return sp.sparse.diags(1 / matrix.sum(axis=1).A1) @ matrix

    def fit_transform(
        self: FreqVectorizer,
        data: List[List[str]],
    ) -> sp.sparse.spmatrix:
        """Return a document-term matrix with frequencies as values. Vocabulary is built
        on the data passed as input.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            sp.sparse.spmatrix: a document-term matrix with frequencies as values.
        """
        self.fit(data)
        return self.transform(data)


class TFIDFVectorizer(CountVectorizer):
    """Create a document-term matrix with TFIDF frequencies as values."""

    def __init__(
        self: TFIDFVectorizer,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        min_count: int = 1,
        smooth: bool = False,
        log_df: bool = False,
    ) -> None:
        """[summary]

        Args:
            self (TFIDFVectorizer): [description]
            max_size (Union[None, int], optional): [description]. Defaults to None.
            min_freq (float, optional): [description]. Defaults to 0.
            max_freq (float, optional): [description]. Defaults to 1.
            smooth (bool, optional): [description]. Defaults to False.
            log_df (bool, optional): [description]. Defaults to False.
        """
        super().__init__(
            max_size=max_size, min_freq=min_freq, max_freq=max_freq, min_count=min_count
        )
        self.smooth = smooth
        self.log_df = log_df

    def fit(
        self: TFIDFVectorizer,
        data: List[List[str]],
    ) -> None:
        """Create the vocabulary.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.
        """
        super().fit(data)

    def transform(self: TFIDFVectorizer, data: List[List[str]]) -> sp.sparse.spmatrix:
        """Return a document-term matrix with TFIDF frequencies as values. Words not in
        vocabulary are discarded.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            sp.sparse.spmatrix: a document-term matrix with TFIDF frequencies
                as values.
        """
        matrix = super().transform(data)
        tf = sp.sparse.diags(1 / matrix.sum(axis=1).A1) @ matrix
        delta = 1 if self.smooth else 0
        if self.log_df:
            idf = np.log(matrix.shape[0] + delta) - np.log(
                (matrix > 0).astype(int).sum(axis=0).A1 + delta
            )
        else:  # linear tfidf
            idf = (matrix.shape[0] + delta) / (matrix > 0).astype(int).sum(axis=0).A1
        return tf @ sp.sparse.diags(idf)

    def fit_transform(
        self: TFIDFVectorizer, data: List[List[str]]
    ) -> sp.sparse.spmatrix:
        """Return a document-term matrix with TFIDF frequencies as values. Vocabulary
        is built on the data passed as input.

        Args:
            data (List[List[str]]): a list of list of tokens. Outer list map to
                documents while inner list map to words in each document. For example
                the output of a :class:`transform.Tokenizer`.

        Returns:
            sp.sparse.spmatrix: a document-term matrix with TFIDF frequencies as
                values.
        """
        self.fit(data)
        return self.transform(data)
