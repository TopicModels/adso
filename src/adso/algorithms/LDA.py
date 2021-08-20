from __future__ import annotations

import os
import re
import subprocess
from shutil import which
from typing import TYPE_CHECKING, Any, Dict, Optional

import dask.array as da
import dask.dataframe as dd
import numpy as np
import sparse
from gensim.models.ldamulticore import LdaMulticore

from .. import common
from ..common import get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data import Dataset


class LDAVB(TMAlgorithm):
    def __init__(self, n: int, **kwargs) -> None:
        self.n = n
        self.kwargs = kwargs

    def fit_transform(self, dataset: Dataset, name: str) -> TopicModel:
        # https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore
        model = LdaMulticore(
            corpus=dataset.get_gensim_corpus(),
            id2word=dataset.get_gensim_vocab(),
            num_topics=self.n,
            random_state=get_seed(),
            **self.kwargs,
        )
        # To get back DT matrix https://github.com/bmabey/pyLDAvis/blob/master/pyLDAvis/gensim_models.py
        topic_word_matrix = model.get_topics()
        doc_topic_matrix = model.inference(dataset.get_gensim_corpus())[0]

        self.model = model
        return TopicModel.from_array(name, topic_word_matrix, doc_topic_matrix)


# https://github.com/RaRe-Technologies/gensim/blob/release-3.8.3/gensim/models/wrappers/ldamallet.py
# https://programminghistorian.org/en/lessons/topic-modeling-and-mallet
# https://stackoverflow.com/questions/42089942/how-to-get-probability-of-words-of-topics-in-mallet
class LDAGS(TMAlgorithm):
    def __init__(
        self,
        n: int,
        memory: Optional[str] = "1G",
        mallet_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n = n
        self.memory = memory
        self.mallet_args = mallet_args

    def fit_transform(self, dataset: Dataset, name: str) -> TopicModel:

        (common.PROJDIR / "mallet" / name).mkdir(exist_ok=True, parents=True)
        doc_topic_path = (
            common.PROJDIR / "mallet" / name / (name + ".doc_topic.mallet.out")
        )
        topic_word_path = (
            common.PROJDIR / "mallet" / name / (name + ".topic_word.mallet.out")
        )

        command = (
            "mallet train-topics "
            + f"--input {str(dataset.get_mallet_corpus())} "
            + f"--num-topics {self.n} "
            + f"--output-doc-topics {str(doc_topic_path)} "
            + f"--topic-word-weights-file {str(topic_word_path)} "
        )

        if get_seed():
            command += f"--random-seed {get_seed()} "
        if os.cpu_count() is not None:
            command += f"--num-threads {os.cpu_count()} "
        if self.mallet_args:
            for key, value in self.mallet_args.items():
                command += f"--{key} {value} "

        mallet_path = which("mallet")
        if mallet_path:
            with open(mallet_path) as f:
                mallet_script = f.read()
            with open(mallet_path, "w") as f:
                if ".bat" in mallet_path:  # win
                    pattern = r"^set MALLET_MEMORY=.*$"
                    subs = f"set MALLET_MEMORY={self.memory}"
                else:
                    pattern = r"^MEMORY=.*$"
                    subs = f"MEMORY={self.memory}"
                f.write(
                    re.sub(
                        pattern,
                        subs,
                        mallet_script,
                        flags=re.M,
                    )
                )
        else:
            raise RuntimeError(
                "MALLET not found, check to be available in PATH (mallet exec and not mallet bin folder). Install it with conda should resolve the issue"
            )

        print(command)
        subprocess.run(command, shell=True, check=True)

        doc_topic_df = (
            dd.read_csv(doc_topic_path, sep="\t", header=None)
            .set_index(1)
            .drop(columns=0)
        )
        doc_topic_matrix = da.zeros((dataset.n_doc(), self.n))
        doc_topic_matrix[
            doc_topic_df.index.to_dask_array().compute(), :
        ] = doc_topic_df.to_dask_array().compute_chunck_size()

        topic_word_df = dd.read_csv(topic_word_path, sep="\t", header=None)
        topic_word_matrix = da.from_array(
            sparse.COO(
                topic_word_df[[0, 1]].values.T,
                data=topic_word_df[2].values,
                fill_value=0,
                shape=(self.n, dataset.n_word()),
            )
        ).map_blocks(lambda b: b.todense(), dtype=np.dtype(float))

        return TopicModel.from_dask_array(
            name,
            topic_word_matrix,
            doc_topic_matrix,
        )
