"""Dataset class.

Define data-container for other classes.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from itertools import chain
from pathlib import Path
from sys import modules
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import sparse
import tomotopy.utils
import zarr
from dask_ml.preprocessing import LabelEncoder
from gensim.corpora.malletcorpus import MalletCorpus
from more_itertools import chunked, unzip

from .. import common
from ..algorithms.vectorizer import Vectorizer
from .corpus import File, Pickled, Raw, WithVocab

try:
    import graph_tool.all as gt

except ImportError:
    print(
        "graph-tool not found, hSBM algorithm not available.\nInstall it with conda (graph-tool package) should resolve the issue"
    )

if TYPE_CHECKING:
    from .corpus import Corpus


class Dataset:
    """Dataset class."""

    def __init__(self, name: str, overwrite: bool = False, load: bool = False) -> None:
        self.name = name
        self.path = common.PROJDIR / self.name
        try:
            self.path.mkdir(exist_ok=overwrite, parents=True)
        except FileExistsError:
            raise RuntimeError(
                "Directory already exist. Allow overwrite or load existing dataset."
            )

        self.vectorizer: Optional[Vectorizer] = None
        self.data: Dict[str, Corpus] = {}
        self.shape: Optional[Tuple[int, int]] = None

        if not load:
            self.save()

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

        dataset = cls(loaded["name"], overwrite=True, load=True)
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
    ) -> Dataset:
        dataset = cls(name, overwrite=overwrite)

        dataset.data["raw"] = Raw.from_dask_array(
            common.PROJDIR / name / (name + ".raw.zarr.zip"),
            da.concatenate(
                [
                    da.from_array(np.array(chunk, dtype=np.bytes_))
                    for chunk in chunked(iterator, batch_size)
                ]
            ),
            overwrite=overwrite,
        )
        dataset.save()
        return dataset

    @classmethod
    def from_array(
        cls,
        name: str,
        array: np.ndarray,
        overwrite: bool = False,
    ) -> Dataset:
        dataset = cls(name, overwrite=overwrite)

        dataset.data["raw"] = Raw.from_array(
            common.PROJDIR / name / (name + ".raw.zarr.zip"),
            array,
            overwrite=overwrite,
        )
        dataset.save()
        return dataset

    @classmethod
    def from_dask_array(
        cls,
        name: str,
        array: da.array,
        overwrite: bool = False,
    ) -> Dataset:
        dataset = cls(name, overwrite=overwrite)

        dataset.data["raw"] = Raw.from_dask_array(
            common.PROJDIR / name / (name + ".raw.zarr.zip"),
            array,
            overwrite=overwrite,
        )
        dataset.save()
        return dataset

    @classmethod
    def from_count_matrix(
        cls,
        name: str,
        count_matrix: da.array,
        vocab: da.array,
        overwrite: bool = False,
    ) -> Dataset:
        dataset = cls(name, overwrite=overwrite)

        assert vocab.shape[0] == count_matrix.shape[1]

        dataset.data["count_matrix"] = WithVocab.from_dask_array(
            dataset.path / (dataset.name + ".count_matrix.zarr.zip"),
            count_matrix,
            vocab,
        )
        dataset.save()
        return dataset

    def get_corpus(self) -> zarr.array:
        return self.data["raw"].get()

    def set_vectorizer_params(
        self,
        # tokenizer: Optional[Callable] = tokenize_and_stem,
        # stop_words: Optional[Iterable[str]] = None,
        # strip_accents: Optional[str] = "unicode",
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        self.vectorizer = Vectorizer(
            self.path / (self.name + ".vectorizer.pickle"),
            overwrite=overwrite,
            **kwargs,
        )
        self.save()

    def _compute_count_matrix(self) -> None:
        if self.vectorizer is None:
            self.set_vectorizer_params()

        self.vectorizer.fit_transform(self)  # type: ignore[union-attr]

    def get_count_matrix(self, sparse: bool = True) -> zarr.array:
        if "count_matrix" not in self.data:
            self._compute_count_matrix()
        return self.data["count_matrix"].get()

    def get_frequency_matrix(self) -> sparse.COO:
        count_matrix = da.array(self.get_count_matrix())
        return (
            (count_matrix / (count_matrix.sum(axis=1)[:, np.newaxis]))
            .map_blocks(lambda x: sparse.COO(x), dtype=np.dtype(float))
            .compute()
        )

    def get_vocab(self) -> zarr.array:
        if "count_matrix" not in self.data:
            self._compute_count_matrix()
        return self.data["count_matrix"].get_vocab()  # type: ignore[attr-defined]

    def get_gensim_corpus(self) -> Iterable[List[Tuple[int, float]]]:
        if "gensim" not in self.data:
            path = self.path / (self.name + ".gensim")
            count_matrix = self.get_count_matrix()
            self.data["gensim"] = Pickled.from_object(
                path,
                [
                    [item for item in enumerate(row.tolist()) if (item[1] != 0)]
                    for row in count_matrix.islice()
                ],
            )
            self.save()
        return self.data["gensim"].get()

    def get_gensim_vocab(self) -> Dict[int, str]:
        return {index[0]: x for (index, x) in np.ndenumerate(self.get_vocab()[...])}

    def get_tomotopy_corpus(self) -> tomotopy.utils.Corpus:
        if "tomotopy" not in self.data:
            path = self.path / (self.name + ".tomotopy")
            corpus = tomotopy.utils.Corpus()
            count_matrix = self.get_count_matrix()
            for row in count_matrix.islice():
                corpus.add_doc(
                    words=list(
                        chain(
                            *[
                                [str(word)] * count
                                for word, count in enumerate(row.tolist())
                                if (count != 0)
                            ]
                        )
                    )
                )
            corpus.save(str(path))
            self.data["tomotopy"] = File(path)
            self.save()
        return tomotopy.utils.Corpus.load(str(self.data["tomotopy"].get()))

    def get_mallet_corpus(self) -> Path:
        if "mallet" not in self.data:
            path = self.path / (self.name + ".mallet")
            command = (
                'mallet import-file --keep-sequence --token-regex "\\p{N}+" '
                + f"--input {self.get_mallet_plain_corpus()} --output {path}"
            )
            print(command)
            subprocess.run(command, shell=True, check=True)
            self.data["mallet"] = File(path)
            self.save()
        return self.data["mallet"].get()

    def get_mallet_plain_corpus(self) -> Path:
        if "mallet_plain" not in self.data:
            path = self.path / (self.name + ".mallet.plain")
            MalletCorpus.save_corpus(
                path,
                self.get_gensim_corpus(),
                # id2word=self.get_gensim_vocab(),
            )
            self.data["mallet_plain"] = File(path)
            self.save()
        return self.data["mallet_plain"].get()

    def get_topicmapping_corpus(self) -> tomotopy.utils.Corpus:
        if "topicmapping" not in self.data:
            path = self.path / (self.name + ".topicmapping")
            count_matrix = self.get_count_matrix()
            with path.open("x") as f:
                for row in count_matrix.islice():
                    f.write(
                        (" ").join(
                            [
                                (" ").join([str(word)] * count)
                                for word, count in enumerate(row.tolist())
                                if (count != 0)
                            ]
                        )
                        + "\n"
                    )
            self.data["topicmapping"] = File(path)
            self.save()
        return self.data["topicmapping"].get()

    def get_shape(self) -> Tuple[int, int]:
        if not self.shape:
            self.shape = self.get_count_matrix().shape
            self.save()
        return self.shape

    def n_doc(self) -> int:
        if "count_matrix" in self.data:
            return self.get_shape()[0]
        else:
            return self.get_corpus().shape[
                0
            ]  # avoid to compute count_matrix if not necessary

    def n_word(self) -> int:
        return self.get_shape()[1]

    if "graph_tool" in modules:

        def _compute_gt_graph(self) -> None:
            path = path = self.path / (self.name + ".gt.gz")
            g = gt.Graph(directed=False)
            name = g.vp["name"] = g.new_vp("int")
            kind = g.vp["kind"] = g.new_vp("int")
            ecount = g.ep["count"] = g.new_ep("int")

            docs_add: defaultdict = defaultdict(lambda: g.add_vertex())
            words_add: defaultdict = defaultdict(lambda: g.add_vertex())

            count_matrix = (
                da.array(self.get_count_matrix())
                .map_blocks(lambda b: sparse.COO(b), dtype=np.dtype(int))
                .compute()
            )

            n_doc, n_word = self.get_shape()

            for i_d in range(n_doc):
                d = docs_add[i_d]
                name[d] = i_d
                kind[d] = 0

            for i_w in range(n_word):
                w = words_add[i_w]
                name[w] = i_w
                kind[w] = 1

            for i in range(count_matrix.nnz):
                i_d, i_w = count_matrix.coords[:, i]
                e = g.add_edge(i_d, n_doc + i_w)
                ecount[e] = count_matrix.data[i]

            g.save(str(path))
            self.data["gt"] = File(path)

        def get_gt_graph_path(self) -> Path:
            if "gt" not in self.data:
                self._compute_gt_graph()
            return self.data["gt"].get()

        def get_gt_graph(self) -> gt.Graph:
            # https://github.com/martingerlach/hSBM_Topicmodel/blob/master/sbmtm.py
            return gt.load_graph(str(self.get_gt_graph_path()))

    # def get_igraph_graph(self) -> igraph.Graph:
    #     if "igraph" not in self.data:
    #         path = self.path / (self.name + ".igraph")
    #         igraph.Graph.Incidence(
    #             self.get_count_matrix().tolist(), weighted="count"
    #         ).write_picklez(path)
    #         self.data["igraph"] = File(path)
    #     return igraph.Graph.Read_Picklez(self.data["igraph"].get())


class LabeledDataset(Dataset):
    def __init__(self, name: str, overwrite: bool = False, load: bool = False) -> None:
        self.labels: Dict[str, Corpus] = {}
        super().__init__(name, overwrite=overwrite, load=load)

    def serialize(self) -> dict:
        save = super().serialize()
        save["labels"] = {key: self.labels[key].serialize() for key in self.labels}
        return save

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> LabeledDataset:

        path = Path(path)
        if path.is_dir():
            with (path / (path.name + ".json")).open(mode="r") as f:
                loaded = json.load(f)
        else:
            with path.open(mode="r") as f:
                loaded = json.load(f)
            path = path.parent

        dataset = cls(loaded["name"], overwrite=True, load=True)
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

        dataset.labels = {
            key: globals()[loaded["labels"][key]["format"]].load(
                Path(loaded["labels"][key]["path"]), loaded["labels"][key]["hash"]
            )
            for key in loaded["labels"]
        }

        dataset.save()
        return dataset

    @classmethod
    def from_iterator(  # type: ignore[override]
        cls,
        name: str,
        iterator: Iterable[Tuple[str, str]],
        batch_size: int = 64,
        overwrite: bool = False,
    ) -> LabeledDataset:
        #
        # An alternative implementation can use itertool.tee + threading/async
        # https://stackoverflow.com/questions/50039223/how-to-execute-two-aggregate-functions-like-sum-concurrently-feeding-them-f
        # https://github.com/omnilib/aioitertools
        #
        # (Label, Doc)

        dataset = cls(name, overwrite=overwrite)
        data_path = common.PROJDIR / name / (name + ".raw.zarr.zip")
        label_path = common.PROJDIR / name / (name + ".label.raw.zarr.zip")

        labels, docs = unzip(iterator)

        if data_path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            dataset.data["raw"] = Raw.from_dask_array(
                data_path,
                da.concatenate(
                    [
                        da.from_array(np.array(chunk))
                        for chunk in chunked(docs, batch_size)
                    ]
                ),
                overwrite=overwrite,
            )

        if label_path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            dataset.labels["raw"] = Raw.from_dask_array(
                label_path,
                da.concatenate(
                    [
                        da.from_array(np.array(chunk))
                        for chunk in chunked(labels, batch_size)
                    ]
                ),
                overwrite=overwrite,
            )

        dataset.save()
        return dataset

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        name: str,
        labels: np.ndarray,
        docs: np.ndarray,
        overwrite: bool = False,
    ) -> LabeledDataset:
        dataset = cls(name, overwrite=overwrite)
        data_path = common.PROJDIR / name / (name + ".raw.zarr.zip")
        label_path = common.PROJDIR / name / (name + ".label.raw.zarr.zip")

        if len(labels) == len(docs):

            if data_path.is_file() and (not overwrite):
                raise RuntimeError("File already exists")
            else:
                dataset.data["raw"] = Raw.from_array(
                    data_path,
                    docs,
                    overwrite=overwrite,
                )

            if label_path.is_file() and (not overwrite):
                raise RuntimeError("File already exists")
            else:
                dataset.labels["raw"] = Raw.from_array(
                    label_path, labels, overwrite=overwrite
                )

        else:
            raise RuntimeError("Different number of elements in labels and docs")

        dataset.save()
        return dataset

    @classmethod
    def from_dask_array(  # type: ignore[override]
        cls,
        name: str,
        labels: da.array,
        docs: da.array,
        overwrite: bool = False,
    ) -> LabeledDataset:
        dataset = cls(name, overwrite=overwrite)
        data_path = common.PROJDIR / name / (name + ".raw.zarr.zip")
        label_path = common.PROJDIR / name / (name + ".label.raw.zarr.zip")

        if len(labels) == len(docs):

            if data_path.is_file() and (not overwrite):
                raise RuntimeError("File already exists")
            else:
                dataset.data["raw"] = Raw.from_dask_array(
                    data_path,
                    docs,
                    overwrite=overwrite,
                )

            if label_path.is_file() and (not overwrite):
                raise RuntimeError("File already exists")
            else:
                dataset.labels["raw"] = Raw.from_dask_array(
                    label_path, labels, overwrite=overwrite
                )

        else:
            raise RuntimeError("Different number of elements in labels and docs")

        dataset.save()
        return dataset

    @classmethod
    def from_count_matrix(  # type: ignore[override]
        cls,
        name: str,
        count_matrix: da.array,
        vocab: da.array,
        labels: da.array,
        overwrite: bool = False,
    ) -> LabeledDataset:
        dataset = cls(name, overwrite=overwrite)

        assert vocab.shape[0] == count_matrix.shape[1]
        assert labels.shape[0] == count_matrix.shape[0]

        dataset.data["count_matrix"] = WithVocab.from_dask_array(
            dataset.path / (dataset.name + ".count_matrix.zarr.zip"),
            count_matrix,
            vocab,
        )
        dataset.labels["raw"] = Raw.from_dask_array(
            dataset.path / (dataset.name + ".label.raw.zarr.zip"),
            labels,
            overwrite=overwrite,
        )
        dataset.save()
        return dataset

    def get_labels(self) -> zarr.array:
        return self.labels["raw"].get()

    def get_labels_vect(self) -> zarr.array:
        if "vect" not in self.labels:
            encoder = LabelEncoder()
            labels = encoder.fit_transform(da.array(self.get_labels()))
            self.labels["vect"] = WithVocab.from_dask_array(
                self.path / (self.name + ".label.vect.zarr.zip"),
                labels,
                encoder.classes_.compute_chunk_sizes(),
            )
            self.save()
        return self.labels["vect"].get()

    def get_labels_matrix(self) -> zarr.array:
        if "matrix" not in self.labels:
            self.labels["matrix"] = Raw.from_dask_array(
                self.path / (self.name + ".label.matrix.zarr.zip"),
                da.array(
                    sparse.COO(
                        [range(self.n_doc()), self.get_labels_vect()],
                        1,
                    )
                ).map_blocks(lambda b: b.todense()),
            )
            self.save()
        return self.labels["matrix"].get()
