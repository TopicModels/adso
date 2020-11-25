"""Vectorizer class."""

from __future__ import annotations

from collections import Counter
from functools import reduce
from itertools import chain, starmap
from typing import Union, List

import numpy as np
import scipy as sp


class Vocab:  # Inspiraed by torchtext.vocab.Vocab
    def __init__(
        self: Vocab,
        count: Counter,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        min_count: int = 1,
    ) -> None:

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

        self.itos = dict(enumerate(words))
        self.stoi = dict(map(reversed, self.itos.items()))

    def __getitem__(self: Vocab, index: Union[int, str]) -> Union[str, int]:
        if isinstance(index, str):
            return self.stoi[index]
        elif isinstance(index, int):
            return self.itos[index]
        else:
            raise ValueError("index must be a string or an int")

    def __len__(self: Vocab) -> int:
        return len(self.stoi)


class Vectorizer:
    def __init__(
        self: Vectorizer,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        min_count: int = 1,
        mode: str = "matrix",
        max_length: int = 100,
    ) -> None:
        if not ((0 <= min_freq <= 1) and (0 <= max_freq <= 1)):
            raise ValueError("min_freq and max_freq must be in the [0,1] interval")
        if mode not in ["matrix", "list"]:
            ValueError("mode not defined (choose matrix or list)")

        self.vocab: Vocab
        self.max_size = max_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_count = min_count
        self.max_length = max_length
        self.mode = mode

    def fit(
        self: Vectorizer,
        data: List[List[str]],
    ) -> None:
        count = reduce(lambda x, y: x + y, map(Counter, data))
        self.vocab = Vocab(
            count, self.max_size, self.min_freq, self.max_freq, self.min_count
        )

    def transform(
        self: Vectorizer, data: List[List[str]]
    ) -> Union[np.array, sp.sparse.spmatrix]:
        if not self.vocab:
            NameError("It is necessary to fit before transform")

        def toindex(s: str) -> Union[None, str]:
            try:
                return self.vocab[s]
            except KeyError:
                return None

        if self.mode == "matrix":

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

        elif self.mode == "list":

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
        self: Vectorizer,
        data: List[List[str]],
    ) -> Union[np.array, sp.sparse.spmatrix]:
        self.fit(data)
        return self.transform(data)


class FreqVectorizer(Vectorizer):
    def __init__(
        self: FreqVectorizer,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
    ) -> None:
        super().__init__(max_size=max_size, min_freq=min_freq, max_freq=max_freq)

    def fit(
        self: FreqVectorizer,
        data: List[List[str]],
    ) -> None:
        super().fit(data)

    def transform(self: FreqVectorizer, data: List[List[str]]) -> sp.sparse.spmatrix:
        matrix = super().transform(data)
        return sp.sparse.diags(1 / matrix.sum(axis=1).A1) @ matrix

    def fit_transform(
        self: FreqVectorizer,
        data: List[List[str]],
    ) -> sp.sparse.spmatrix:
        self.fit(data)
        return self.transform(data)


class TFIDFVectorizer(Vectorizer):
    def __init__(
        self: TFIDFVectorizer,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        smooth: bool = False,
        log_df: bool = False,
    ) -> None:
        super().__init__(max_size=max_size, min_freq=min_freq, max_freq=max_freq)
        self.smooth = smooth
        self.lod_df = log_df

    def fit(
        self: TFIDFVectorizer,
        data: List[List[str]],
    ) -> None:
        super().fit(data)

    def transform(self: TFIDFVectorizer, data: List[List[str]]) -> sp.sparse.spmatrix:
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
        self.fit(data)
        return self.transform(data)
