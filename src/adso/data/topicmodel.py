from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

import dask.array as da
import numpy as np

from .. import common
from ..data.corpus import Raw, Sparse

if TYPE_CHECKING:
    from ..data.corpus import Corpus


class TopicModel:
    def __init__(self, name: str, overwrite: bool = False) -> None:
        self.name = name
        self.path = common.PROJDIR / self.name
        try:
            self.path.mkdir(exist_ok=overwrite, parents=True)
        except FileExistsError:
            raise RuntimeError(
                "Directory already exist. Allow overwrite or load existing dataset."
            )

        self.data: Dict[str, Corpus] = {}

        self.save()

    def serialize(self) -> dict:
        save: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
        }
        save["data"] = {key: self.data[key].serialize() for key in self.data}
        return save

    def save(self) -> None:
        with (self.path / (self.name + ".json")).open(mode="w") as f:
            json.dump(self.serialize(), f, indent=4)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "TopicModel":
        path = Path(path)
        if path.is_dir():
            with (path / (path.name + ".json")).open(mode="r") as f:
                loaded = json.load(f)
        else:
            with path.open(mode="r") as f:
                loaded = json.load(f)
            path = path.parent

        model = cls(loaded["name"], overwrite=True)
        model.path = path

        model.data = {
            key: globals()[loaded["data"][key]["format"]].load(
                Path(loaded["data"][key]["path"]), loaded["data"][key]["hash"]
            )
            for key in loaded["data"]
        }

        model.save()
        return model

    @classmethod
    def from_dask_array(
        cls,
        name: str,
        topic_word_matrix: da.array,
        doc_topic_matrix: da.array,
        overwrite: bool = False,
    ) -> "TopicModel":
        model = cls(name, overwrite=overwrite)
        model.data["topic_word"] = Sparse.from_dask_array(
            model.path / "topic_word.hdf5", topic_word_matrix, overwrite=overwrite
        )
        model.data["doc_topic"] = Sparse.from_dask_array(
            model.path / "doc_topic.hdf5", doc_topic_matrix, overwrite=overwrite
        )
        model.save()
        return model

    def get_topic_word_matrix(
        self,
        skip_hash_check: bool = False,
        sparse: bool = False,
        normalize: bool = False,
    ) -> da.array:
        topic_word = self.data["topic_word"].get(sparse=sparse)  # type: ignore[call-arg]
        if normalize:
            if sparse:
                return (
                    topic_word
                    / (topic_word.sum(axis=1)).map_blocks(
                        lambda b: b.todense(), dtype=np.float64
                    )[:, np.newaxis]
                )
            else:
                return topic_word / (topic_word.sum(axis=1))[:, np.newaxis]
        return topic_word

    def get_doc_topic_matrix(
        self,
        skip_hash_check: bool = False,
        sparse: bool = False,
        normalize: bool = False,
    ) -> da.array:
        doc_topic = self.data["doc_topic"].get(sparse=sparse)  # type: ignore[call-arg]
        if normalize:
            if sparse:
                return (
                    doc_topic
                    / (doc_topic.sum(axis=1)).map_blocks(
                        lambda b: b.todense(), dtype=np.float64
                    )[:, np.newaxis]
                )
            else:
                return doc_topic / (doc_topic.sum(axis=1))[:, np.newaxis]
        return doc_topic

    def get_labels(self) -> da.array:
        if "labels" not in self.data:
            self.data["labels"] = Raw.from_dask_array(
                self.path / "labels.hdf5",
                da.argmax(self.get_doc_topic_matrix(sparse=False), axis=1),
            )
            self.save()
        return self.data["labels"].get()
