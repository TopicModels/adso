"""Dataset class.

Define data-container for other classes.
"""

import json
import os
import pickle
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import dask.array as da
import dask.bag as db
import dask_ml
import numpy as np
import sparse
from more_itertools import chunked

from .. import common as adso_common
from .common import nltk_download, tokenize_and_stem
from .corpus import Corpus, CountMatrix, Raw
from ..algorithm.vectorizer import Vectorizer


class Dataset:
    """Dataset class."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = adso_common.PROJDIR / self.name
        # maybe add existence check?
        self.path.mkdir(exist_ok=True, parents=True)

        self.vectorizer: Optional[Vectorizer] = None
        self.data: Dict[str, Corpus] = {}

    def serialize(self) -> dict:
        save: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
        }
        if self.vectorizer is not None:
            save["vectorizer"] = self.vectorizer.serialize()
        save["data"] = {key: self.data[key].serialize() for key in self.data}
        return save

    def save(self) -> None:
        with (self.path / (self.name + ".json")).open(mode="w") as f:
            json.dump(self.serialize(), f, indent=4)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Dataset":
        path = Path(path)
        if path.is_dir():
            with (path / (path.name + ".json")).open(mode="r") as f:
                loaded = json.load(f)
        else:
            with path.open(mode="r") as f:
                loaded = json.load(f)
            path = path.parent

        dataset = cls(loaded["name"])
        dataset.path = path

        if "vectorizer" in loaded:
            dataset.vectorizer = Vectorizer.load(
                loaded["vectorizer"]["path"], loaded["vectorizer"]["hash"]
            )  # type: ignore[assignment]
        else:
            dataset.vectorizer = None

        dataset.data = {
            key: globals()[loaded["data"][key]["format"]].load(
                Path(loaded["data"][key]["path"]), loaded["data"][key]["hash"]
            )
            for key in loaded["data"]
        }

        dataset.save()
        return dataset

    @classmethod
    def from_iterator(
        cls,
        name: str,
        iterator: Iterable[str],
        batch_size: int = 64,
        overwrite: bool = False,
    ) -> "Dataset":
        dataset = cls(name)

        dataset.data["raw"] = Raw.from_dask_array(
            adso_common.PROJDIR / name / (name + ".raw.hdf5"),
            da.concatenate(
                [
                    da.from_array(np.array(chunk, dtype=np.dtype(bytes)))
                    for chunk in chunked(iterator, batch_size)
                ]
            ),
            overwrite=overwrite,
        )
        dataset.save()
        return dataset

    def get_corpus(self) -> da.array:
        return self.data["raw"].get()

    def set_vectorizer_params(
        self,
        tokenizer: Optional[Callable] = tokenize_and_stem,
        stop_words: Optional[Iterable[str]] = None,
        strip_accents: Optional[str] = "unicode",
        overwrite: bool = False,
        **kwargs
    ) -> None:
        if tokenizer == tokenize_and_stem:
            nltk_download("punkt")

        # could be necessary to tokenize the stopwords
        if (stop_words is not None) and (tokenizer is not None):
            stop_words = set(chain.from_iterable([tokenizer(sw) for sw in stop_words]))

        mode = "wb" if overwrite else "xb"

        path = self.path / (self.name + ".vectorizer.pickle")
        pickle.dump(
            dask_ml.feature_extraction.text.CountVectorizer(
                tokenizer=tokenizer, stop_words=stop_words, strip_accents="unicode"
            ),
            path.open(mode),
        )
        self.vectorizer = Vectorizer(path)
        self.save()

    def _compute_count_matrix(self) -> None:
        if self.vectorizer is None:
            self.set_vectorizer_params()

        vectorizer = self.vectorizer.get()  # type: ignore[union-attr]
        corpus = self.data["raw"].get()
        bag = db.from_sequence([doc.compute().item() for doc in corpus])

        count_matrix = (
            vectorizer.fit_transform(bag)
            .map_blocks(lambda x: sparse.COO(x).todense())
            .compute_chunk_sizes()
        )

        vocab = da.from_array(
            np.array(vectorizer.get_feature_names(), dtype=np.dtype(bytes))
        )

        self.data["count_matrix"] = CountMatrix.from_dask_array(
            self.path / (self.name + ".count_matrix"), count_matrix, vocab
        )

        pickle.dump(
            vectorizer,
            self.vectorizer.path.open("wb"),  # type: ignore[union-attr]
        )
        self.vectorizer.update_hash()  # type: ignore[union-attr]

        self.save()

    def get_count_matrix(self) -> da.array:
        if "count_matrix" not in self.data:
            self._compute_count_matrix()
        return self.data["count_matrix"].get()

    def get_vocab(self) -> da.array:
        if "count_matrix" not in self.data:
            self._compute_count_matrix()
        return self.data["count_matrix"].get_vocab()  # type: ignore[attr-defined]


class LabeledDataset(Dataset):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.labels: Dict[str, Corpus] = {}

    def serialize(self) -> dict:
        save = super().serialize()
        save["labels"] = {key: self.labels[key].serialize() for key in self.labels}
        return save

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "LabeledDataset":

        path = Path(path)
        if path.is_dir():
            with (path / (path.name + ".json")).open(mode="r") as f:
                loaded = json.load(f)
        else:
            with path.open(mode="r") as f:
                loaded = json.load(f)
            path = path.parent

        dataset = cls(loaded["name"])
        dataset.path = path

        dataset.data = {
            key: globals()[loaded["data"][key]["format"]].load(
                Path(loaded["data"][key]["path"]), loaded["data"][key]["hash"]
            )
            for key in loaded["data"]
        }

        dataset.labels = {
            key: globals()[loaded["labels"][key]["format"]].load(
                Path(loaded["labels"][key]["path"]), loaded["labels"][key]["hash"]
            )
            for key in loaded["labels"]
        }

        dataset.save()
        return dataset

    @classmethod
    def from_iterator(cls, name: str, iterator: Iterable[Tuple[str, str]], batch_size: int = 64, overwrite: bool = False) -> "LabeledDataset":  # type: ignore[override]
        #
        # An alternative implementation can use itertool.tee + threading/async
        # https://stackoverflow.com/questions/50039223/how-to-execute-two-aggregate-functions-like-sum-concurrently-feeding-them-f
        # https://github.com/omnilib/aioitertools
        #
        # (Label, Doc)

        dataset = cls(name)
        data_path = adso_common.PROJDIR / name / (name + ".raw.hdf5")
        label_path = adso_common.PROJDIR / name / (name + ".label.raw.hdf5")
        data = da.concatenate(
            [
                da.from_array(np.array(chunk, dtype=np.dtype(bytes)))
                for chunk in chunked(iterator, batch_size)
            ]
        )

        if data_path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            data[:, 1].squeeze().to_hdf5(data_path, "/raw", shuffle=False)
            dataset.data["raw"] = Raw(data_path)

        if label_path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            data[:, 0].squeeze().to_hdf5(label_path, "/raw", shuffle=False)
            dataset.labels["raw"] = Raw(label_path)

        dataset.save()
        return dataset

    def get_labels(self) -> da.array:
        return self.labels["raw"].get()
