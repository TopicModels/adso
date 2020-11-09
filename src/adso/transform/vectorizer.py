"""Vectorizer class."""

from __future__ import annotations

from collections import Counter
from functools import reduce
from itertools import chain, starmap
from typing import Union

import torch

from ..data import Dataset


class Vocab:  # Inspiraed from torchtext.vocab.Vocab
    def __init__(
        self: Vocab,
        count: Counter,
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
    ) -> None:

        self.min_freq = min_freq
        self.max_freq = max_freq

        total_count = sum(count.values())
        min_freq = self.min_freq * total_count
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
    def __init__(self: Vectorizer) -> None:
        self.vocab: Vocab

    def fit(
        self: Vectorizer,
        data: list(list(str)),
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        max_length: int = 100,
    ) -> None:
        if not ((0 <= min_freq <= 1) and (0 <= max_freq <= 1)):
            raise ValueError("min_freq and max_freq must be in the [0,1] interval")

        self.max_length = max_length

        count = reduce(lambda x, y: x + y, map(Counter, data))
        self.vocab = Vocab(count, max_size, min_freq, max_freq)

    def transform(
        self: Vectorizer, data: list(list(str)), mode: str = "matrix"
    ) -> torch.Tensor:
        if not self.vocab:
            NameError("It is necessary to fit before transform")

        def toindex(s: str) -> Union[None, str]:
            try:
                return self.vocab[s]
            except KeyError:
                return None

        if mode == "matrix":

            def document_to_count(document: list(str)):
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

            tensor = torch.tensor(
                list(
                    chain.from_iterable(
                        starmap(
                            lambda row, doc: list(
                                starmap(lambda col, val: (row, col, val), doc)
                            ),
                            enumerate(map(document_to_count, data)),
                        )
                    )
                )
            )

            return torch.sparse.LongTensor(
                tensor[:, [0, 1]].t(),
                tensor[:, 2],
                torch.Size([len(data), len(self.vocab)]),
            )

        elif mode == "list":

            def document_to_list(document: list(str)):
                return list(
                    filter(
                        lambda tkn: tkn is not None,
                        map(toindex, document),
                    )
                )

            data = list(map(document_to_list, data))
            length = min(max(map(len, data)), self.max_length)

            def pad(data, length):
                if len(data) > length:
                    return data[:length]
                else:
                    return data + [-1] * (length - len(data))

            return torch.LongTensor(list(map(lambda row: pad(row, length), data)))

        else:
            ValueError("mode not defined (choose matrix or list)")

    def fit_transform(
        self: Vectorizer,
        data: list(list(str)),
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        mode: str = "matrix",
        max_length: int = 100,
    ) -> torch.Tensor:
        self.fit(data, max_size, min_freq, max_freq, max_length)
        return self.transform(data, mode)


class FreqVectorizer(Vectorizer):
    def fit(
        self: FreqVectorizer,
        data: list(list(str)),
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
    ) -> None:
        super().fit(data, max_size, min_freq, max_freq)

    def transform(self: FreqVectorizer, data: list(list(str))) -> torch.Tensor:
        tensor = super().transform(data, mode="matrix").to_dense()
        return tensor / torch.sum(tensor, 1, keepdim=True)

    def fit_transform(
        self: FreqVectorizer,
        data: list(list(str)),
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        max_length: int = 100,
    ) -> torch.Tensor:
        self.fit(data, max_size, min_freq, max_freq, max_length)
        return self.transform(data)


class TFIDFVectorizer(Vectorizer):
    def fit(
        self: TFIDFVectorizer,
        data: list(list(str)),
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        smooth: bool = False,
    ) -> None:
        super().fit(data, max_size, min_freq, max_freq)
        self.smooth = smooth

    def transform(self: TFIDFVectorizer, data: list(list(str))) -> torch.Tensor:
        tensor = super().transform(data, mode="matrix").to_dense()
        if self.smooth:  # avoid inf
            return (tensor / torch.sum(tensor, 1, keepdim=True)) * torch.log(
                (tensor.size()[0] + 1)
                / ((tensor / torch.sum(tensor, 0, keepdim=True) + 1))
            )
        else:
            return (tensor / torch.sum(tensor, 1, keepdim=True)) * torch.log(
                tensor.size()[0] / (tensor / torch.sum(tensor, 0, keepdim=True))
            )

    def fit_transform(
        self: TFIDFVectorizer,
        data: list(list(str)),
        max_size: Union[None, int] = None,
        min_freq: float = 0,
        max_freq: float = 1,
        smooth: bool = False,
    ) -> torch.Tensor:
        self.fit(data, max_size, min_freq, max_freq, smooth)
        return self.transform(data)
