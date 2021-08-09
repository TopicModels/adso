from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple, Union

import dask.array as da
import dill
import numpy as np
import sparse
import zarr

from .. import common
from ..data.corpus import Raw

if TYPE_CHECKING:
    from ..data.corpus import Corpus

try:
    from ._sbmtm import sbmtm
except ImportError as e:
    print(e)
    print(
        "graph-tool not found, hSBM algorithm not available.\nInstall it with conda (graph-tool package) should resolve the issue"
    )


class TopicModel:
    def __init__(self, name: str, overwrite: bool = False, load: bool = False) -> None:
        self.name = name
        self.path = common.PROJDIR / self.name
        try:
            self.path.mkdir(exist_ok=overwrite, parents=True)
        except FileExistsError:
            raise RuntimeError(
                "Directory already exist. Allow overwrite or load existing dataset."
            )

        self.data: Dict[str, Corpus] = {}

        if not load:
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

        model = cls(loaded["name"], overwrite=True, load=True)
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
        topic_word_matrix: np.ndarray,
        doc_topic_matrix: np.ndarray,
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
            return zarr.array(topic_word / (topic_word[:].sum(axis=1))[:, np.newaxis])
        return topic_word

    def get_doc_topic_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
    ) -> zarr.array:
        doc_topic = self.data["doc_topic"].get()
        if normalize:
            return zarr.array(doc_topic / (doc_topic[:].sum(axis=1))[:, np.newaxis])
        return doc_topic

    def get_labels(self) -> zarr.array:
        if "labels" not in self.data:
            self.data["labels"] = Raw.from_dask_array(
                self.path / "labels.zarr.zip",
                da.argmax(da.array(self.get_doc_topic_matrix()), axis=1),
            )
            self.save()
        return self.data["labels"].get()


class HierarchicalTopicModel(TopicModel):
    def __init__(self, name: str, overwrite: bool = False, load: bool = False) -> None:
        self.name = name
        self.path = common.PROJDIR / self.name
        try:
            self.path.mkdir(exist_ok=overwrite, parents=True)
        except FileExistsError:
            raise RuntimeError(
                "Directory already exist. Allow overwrite or load existing dataset."
            )

        self.data: Dict[int, Dict[str, Corpus]] = {}  # type: ignore[assignment]
        if not load:
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
    def load(cls, path: Union[str, os.PathLike]) -> "HierarchicalTopicModel":
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
            int(l): {
                key: globals()[loaded["data"][l][key]["format"]].load(
                    Path(loaded["data"][l][key]["path"]), loaded["data"][l][key]["hash"]
                )
                for key in loaded["data"][l]
            }
            for l in loaded["data"]
        }

        model.save()
        return model

    @classmethod
    def from_dask_array(  # type: ignore[override]
        cls,
        name: str,
        matrices: Iterable[Tuple[da.array, da.array]],  # topic_word, doc_topic
        overwrite: bool = False,
    ) -> "HierarchicalTopicModel":
        model = cls(name, overwrite=overwrite)
        for i, mat in enumerate(matrices):
            model.data[i] = {}
            model.data[i]["topic_word"] = Raw.from_dask_array(
                model.path / f"topic_word{i}.zarr.zip",
                mat[0],
                overwrite=overwrite,
            )
            model.data[i]["doc_topic"] = Raw.from_dask_array(
                model.path / f"doc_topic{i}.zarr.zip",
                mat[1],
                overwrite=overwrite,
            )
        model.save()
        return model

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        name: str,
        matrices: List[Tuple[np.ndarray, np.ndarray]],  # topic_word, doc_topic
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
            return zarr.array(topic_word / (topic_word[:].sum(axis=1))[:, np.newaxis])
        return topic_word

    def get_doc_topic_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
        l: int = 0,
    ) -> zarr.array:
        doc_topic = self.data[l]["doc_topic"].get()
        if normalize:
            return zarr.array(doc_topic / (doc_topic[:].sum(axis=1))[:, np.newaxis])
        return doc_topic

    def get_labels(self, l: int = 0) -> zarr.array:
        if "labels" not in self.data[l]:
            self.data[l]["labels"] = Raw.from_array(
                self.path / f"labels{l}.zarr.zip",
                np.argmax(self.get_doc_topic_matrix(l=l), axis=1),  # type: ignore[arg-type]
            )
            self.save()
        return self.data[l]["labels"].get()

    def __getitem__(self, l: int) -> "PseudoTopicModel":
        return PseudoTopicModel(self, l)

    def get_doc_cluster_matrix(self, l: int = 0) -> da.array:
        if "cluster" not in self.data[l]:
            path = common.PROJDIR / "hSBM" / self.name / "model.pkl"
            if path.is_file():
                model = dill.load(path.open("rb"))
            else:
                raise RuntimeError("File not found")

            def get_clusters(model: "sbmtm", l: int = 0) -> da.array:
                # rewrite from _sbmtm to use dask
                D = model.get_D()

                g = model.g
                state = model.state
                state_l = state.project_level(l).copy(overlap=True)
                state_l_edges = state_l.get_edge_blocks()  # labeled half-edges

                # count labeled half-edges, group-memberships
                B = state_l.get_B()

                id_d = np.zeros(g.edge_index_range, dtype=np.dtype(int))
                id_b = np.zeros(g.edge_index_range, dtype=np.dtype(int))
                weig = np.zeros(g.edge_index_range, dtype=np.dtype(int))

                for i, e in enumerate(g.edges()):
                    id_b[i], _ = state_l_edges[e]
                    id_d[i] = int(e.source())
                    weig[i] = g.ep["count"][e]

                n_db = sparse.COO(
                    [id_d, id_b], weig, shape=(D, B), fill_value=0
                )  # number of half-edges incident on document-node d and labeled as cluster

                del weig
                del id_b
                del id_d

                #####
                ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
                n_db = n_db[:, ind_d]
                del ind_d

                # Mixture of clusters into documetns P(d | c)
                p_td_d = n_db / np.sum(n_db, axis=0).todense()[np.newaxis, :]

                return da.array(p_td_d).map_blocks(
                    lambda b: b.todense(), dtype=np.dtype(float)
                )

            self.data[l]["cluser"] = Raw.from_dask_array(
                self.path / f"clusters{l}.zarr.zip", get_clusters(model, l)
            )

        self.data[l]["cluser"].get()

    def get_cluster_labels(self, l: int = 0) -> zarr.array:
        if "cluster_labels" not in self.data[l]:
            self.data[l]["cluster_labels"] = Raw.from_array(
                self.path / f"cluster_labels{l}.zarr.zip",
                np.argmax(self.get_doc_cluster_matrix(l=l), axis=1),  # type: ignore[arg-type]
            )
            self.save()
        return self.data[l]["cluster_labels"].get()


class PseudoTopicModel(TopicModel):
    def __init__(self, parent: HierarchicalTopicModel, idx: int) -> None:
        self.parent = parent
        self.idx = idx

    def save(self) -> None:
        self.parent.save()

    @classmethod
    def from_dask_array(
        cls,
        name: str,
        topic_word_matrix: da.array,
        doc_topic_matrix: da.array,
        overwrite: bool = False,
    ) -> "PseudoTopicModel":
        raise NotImplementedError

    @classmethod
    def from_array(
        cls,
        name: str,
        topic_word_matrix: np.ndarray,
        doc_topic_matrix: np.ndarray,
        overwrite: bool = False,
    ) -> "PseudoTopicModel":
        raise NotImplementedError

    def get_topic_word_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
    ) -> zarr.array:
        return self.parent.get_topic_word_matrix(
            skip_hash_check=skip_hash_check, normalize=normalize, l=self.idx
        )

    def get_doc_topic_matrix(
        self,
        skip_hash_check: bool = False,
        normalize: bool = False,
    ) -> zarr.array:
        return self.parent.get_doc_topic_matrix(
            skip_hash_check=skip_hash_check, normalize=normalize, l=self.idx
        )

    def get_labels(self) -> zarr.array:
        return self.parent.get_labels(l=self.idx)
