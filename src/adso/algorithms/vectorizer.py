from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import dask.array as da
import dask.bag as db
import dill
import numpy as np
import sparse
from dask_ml.feature_extraction.text import CountVectorizer

# from ..data.common import nltk_download, tokenize_and_stem
from ..common import compute_hash
from ..data.corpus import WithVocab
from .common import Algorithm

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class Vectorizer(Algorithm):
    def __init__(
        self,
        path: Path,
        overwrite: bool = False,
        tokenizer: Optional[Callable] = None,
        stop_words: Optional[Iterable[str]] = None,
        strip_accents: Optional[str] = "unicode",
        load: bool = False,
        **kwargs
    ) -> None:
        self.path = path
        if load:
            self.update_hash()
        else:
            # if tokenizer == tokenize_and_stem:
            #    nltk_download("punkt")
            if (stop_words is not None) and (tokenizer is not None):
                stop_words = set(
                    chain.from_iterable([tokenizer(sw) for sw in stop_words])
                )
            model = CountVectorizer(
                tokenizer=tokenizer, stop_words=stop_words, strip_accents=strip_accents
            )
            if path.is_file() and not overwrite:
                raise RuntimeError("File already exists")
            else:
                self.save(model)
                self.update_hash()

    @classmethod
    def load(cls, path: Union[Path, str], hash: Optional[str]) -> Vectorizer:
        path = Path(path)
        if path.is_file():
            vect = cls(path, load=True)
            if (vect.hash == hash) or (hash is None):
                return vect
            else:
                raise RuntimeError("Different hash")
        else:
            raise RuntimeError("File doesn't exists")

    def save(self, model: CountVectorizer) -> None:  # type: ignore[override]
        dill.dump(
            model,
            self.path.open("wb"),
        )
        self.update_hash()

    def get(self) -> CountVectorizer:
        if self.hash == compute_hash(self.path):
            return dill.load(self.path.open("rb"))
        else:
            raise RuntimeError("Different hash")

    def fit_transform(self, dataset: Dataset, update: bool = True) -> None:

        # actually, the list comprehension is a bottleneck
        bag = db.from_sequence([doc.item() for doc in dataset.get_corpus()])
        model = self.get()

        model.fit(bag)

        count_matrix = (
            (model.transform(bag))
            .map_blocks(lambda x: sparse.COO(x, fill_value=0))
            .compute_chunk_sizes()
            .rechunk()
        )

        vocab = da.from_array(np.array(model.get_feature_names()))

        dataset.data["count_matrix"] = WithVocab.from_dask_array(
            dataset.path / (dataset.name + ".count_matrix.zarr.zip"),
            count_matrix,
            vocab,
        )

        if update:
            self.save(model)

        dataset.save()
