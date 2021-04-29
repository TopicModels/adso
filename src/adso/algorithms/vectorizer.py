import pickle
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import dask.array as da
import dask.bag as db
import numpy as np
import sparse
from dask_ml.feature_extraction.text import CountVectorizer

# from ..data.common import nltk_download, tokenize_and_stem
from ..common import compute_hash
from ..data.corpus import CountMatrix
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
        **kwargs
    ) -> None:
        self.path = path
        # if tokenizer == tokenize_and_stem:
        #    nltk_download("punkt")
        if (stop_words is not None) and (tokenizer is not None):
            stop_words = set(chain.from_iterable([tokenizer(sw) for sw in stop_words]))
        model = CountVectorizer(
            tokenizer=tokenizer, stop_words=stop_words, strip_accents=strip_accents
        )
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            self.save(model)

    def save(self, model: CountVectorizer) -> None:  # type: ignore[override]
        pickle.dump(
            model,
            self.path.open("wb"),
        )
        self.update_hash()

    def get(self) -> CountVectorizer:
        if self.hash == compute_hash(self.path):
            return pickle.load(self.path.open("rb"))
        else:
            raise RuntimeError("Different hash")

    def fit_transform(self, dataset: "Dataset", update: bool = True) -> None:

        bag = db.from_sequence([doc.compute().item() for doc in dataset.get_corpus()])
        model = self.get()

        count_matrix = (
            model.fit_transform(bag)
            .map_blocks(lambda x: sparse.COO(x, fill_value=0).todense())
            .compute_chunk_sizes()
        )

        vocab = da.from_array(np.array(model.get_feature_names()))

        dataset.data["count_matrix"] = CountMatrix.from_dask_array(
            dataset.path / (dataset.name + ".count_matrix.hdf5"), count_matrix, vocab
        )

        if update:
            self.save(model)

        dataset.save()
