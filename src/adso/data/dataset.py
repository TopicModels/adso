"""Dataset class.

Define data-container for other classes.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import dask.array as da
import dask_ml
import numpy as np
from more_itertools import chunked

from .. import common as adso_common
from .common import get_nltk_stopwords, tokenize_and_stem
from .corpus import Corpus, Raw
from .vectorizer import Vectorizer


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
        cls, name: str, iterator: Iterable[str], batch_size: int = 64
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
        )
        dataset.save()
        return dataset

    def get_corpus(self) -> da.array:
        return self.data["raw"].get()

    def set_vectorizer_params(
        self,
        tokenizer: Optional[Callable] = tokenize_and_stem,
        stop_words: Optional[Iterable[str]] = get_nltk_stopwords(),
        strip_accents: Optional[str] = "unicode",
        **kwargs
    ) -> None:
        path = self.path / (self.name + ".vectorizer.pickle")
        pickle.dump(
            dask_ml.feature_extraction.text.CountVectorizer(
                tokenizer=tokenizer, stop_words=stop_words, strip_accents="unicode"
            ),
            path.open("xb"),
        )
        self.vectorizer = Vectorizer(path)

    def _compute_count_matrix(self) -> None:
        if self.vectorizer is None:
            self.set_vectorizer_params()
        vectorizer = self.vectorizer.get()  # type: ignore[union-attr]
        corpus = self.data["raw"].get()

        self.data["count_matrix"] = Raw.from_dask_array(
            self.path / (self.name + "count_matrix.hdf5"),
            vectorizer.fit_transform(corpus),
        )

        self.data["vocab"] = Raw.from_dask_array(
            self.path / (self.name + "vocab.hdf5"), vectorizer.get_feature_names()
        )

    def get_count_matrix(self) -> da.array:
        if "count_matrix" not in self.data:
            self._compute_count_matrix()
        return self.data["count_matrix"].get()

    def get_vocab(self) -> da.array:
        if "vocab" not in self.data:
            self._compute_count_matrix()
        return self.data["vocab"].get()


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
    def from_iterator(cls, name: str, iterator: Iterable[Tuple[str, str]], batch_size: int = 64) -> "LabeledDataset":  # type: ignore[override]
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

        if data_path.is_file():
            raise RuntimeError
        else:
            data[:, 1].squeeze().to_hdf5(data_path, "/raw", shuffle=False)
            dataset.data["raw"] = Raw(data_path)

        if label_path.is_file():
            raise RuntimeError
        else:
            data[:, 0].squeeze().to_hdf5(label_path, "/raw", shuffle=False)
            dataset.labels["raw"] = Raw(label_path)

        dataset.save()
        return dataset

    def get_labels(self) -> da.array:
        return self.labels["raw"].get()
