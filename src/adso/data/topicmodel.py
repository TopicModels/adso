from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import zarr

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
        model.data["topic_word"] = Raw.from_dask_array(
            model.path / "topic_word.zarr.zip", topic_word_matrix, overwrite=overwrite
        )
        model.data["doc_topic"] = Raw.from_dask_array(
            model.path / "doc_topic.zarr.zip", doc_topic_matrix, overwrite=overwrite
        )
        model.save()
        return model

    @classmethod
    def from_array(
        cls,
        name: str,
        topic_word_matrix: np.array,
        doc_topic_matrix: np.array,
        overwrite: bool = False,
    ) -> "TopicModel":
        model = cls(name, overwrite=overwrite)
        model.data["topic_word"] = Raw.from_array(
            model.path / "topic_word.zarr.zip", topic_word_matrix, overwrite=overwrite
        )
        model.data["doc_topic"] = Raw.from_array(
            model.path / "doc_topic.zarr.zip", doc_topic_matrix, overwrite=overwrite
        )
        model.save()
        return model

    def get_topic_word_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
    ) -> zarr.array:
        topic_word = self.data["topic_word"].get()
        if normalize:
            return topic_word / (topic_word.sum(axis=1))[:, np.newaxis]
        return topic_word

    def get_doc_topic_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
    ) -> zarr.array:
        doc_topic = self.data["doc_topic"].get()
        if normalize:
            return doc_topic / (doc_topic.sum(axis=1))[:, np.newaxis]
        return doc_topic

    def get_labels(self) -> zarr.array:
        if "labels" not in self.data:
            self.data["labels"] = Raw.from_dask_array(
                self.path / "labels.zarr.zip",
                da.argmax(self.get_doc_topic_matrix(sparse=False), axis=1),
            )
            self.save()
        return self.data["labels"].get()


class HierarchicalTopicModel(TopicModel):
    # TODO: review in order to get a TopicModel-like with [l]
    def __init__(self, name: str, overwrite: bool = False) -> None:
        self.name = name
        self.path = common.PROJDIR / self.name
        try:
            self.path.mkdir(exist_ok=overwrite, parents=True)
        except FileExistsError:
            raise RuntimeError(
                "Directory already exist. Allow overwrite or load existing dataset."
            )

        self.data: Dict[int, Dict[str, Corpus]] = {}  # type: ignore[assignment]
        self.save()

    def serialize(self) -> dict:
        save: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
        }
        save["data"] = {}
        for i in self.data:
            save["data"][i] = {
                key: self.data[i][key].serialize() for key in self.data[i]
            }
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
            key: globals()[loaded["data"][l][key]["format"]].load(
                Path(loaded["data"][l][key]["path"]), loaded["data"][l][key]["hash"]
            )
            for l in loaded["data"]
            for key in loaded["data"][l]
        }

        model.save()
        return model

    @classmethod
    def from_dask_array(  # type: ignore[override]
        cls,
        name: str,
        matrices: List[Tuple[da.array, da.array]],  # topic_word, doc_topic
        overwrite: bool = False,
    ) -> "HierarchicalTopicModel":
        model = cls(name, overwrite=overwrite)
        for i in range(len(matrices)):
            model.data[i] = {}
            model.data[i]["topic_word"] = Sparse.from_dask_array(
                model.path / f"topic_word{i}.zarr.zip",
                matrices[i][0],
                overwrite=overwrite,
            )
            model.data[i]["doc_topic"] = Sparse.from_dask_array(
                model.path / f"doc_topic{i}.zarr.zip",
                matrices[i][1],
                overwrite=overwrite,
            )
        model.save()
        return model

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        name: str,
        matrices: List[Tuple[np.array, np.array]],  # topic_word, doc_topic
        overwrite: bool = False,
    ) -> "HierarchicalTopicModel":
        model = cls(name, overwrite=overwrite)
        for i in range(len(matrices)):
            model.data[i] = {}
            model.data[i]["topic_word"] = Raw.from_array(
                model.path / f"topic_word{i}.zarr.zip",
                matrices[i][0],
                overwrite=overwrite,
            )
            model.data[i]["doc_topic"] = Raw.from_array(
                model.path / f"doc_topic{i}.zarr.zip",
                matrices[i][1],
                overwrite=overwrite,
            )
        model.save()
        return model

    def get_topic_word_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
        l: int = 0,
    ) -> da.array:
        topic_word = self.data[l]["topic_word"].get()
        if normalize:
            return topic_word / (topic_word.sum(axis=1))[:, np.newaxis]
        return topic_word

    def get_doc_topic_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
        l: int = 0,
    ) -> zarr.array:
        doc_topic = self.data[l]["doc_topic"].get()
        if normalize:
            return doc_topic / (doc_topic.sum(axis=1))[:, np.newaxis]
        return doc_topic

    def get_labels(self, l: int = 0) -> zarr.array:
        if "labels" not in self.data[l]:
            self.data[l]["labels"] = Raw.from_array(
                self.path / f"labels{l}.zarr.zip",
                np.argmax(self.get_doc_topic_matrix(l=l), axis=1),
            )
            self.save()
        return self.data[l]["labels"].get()
