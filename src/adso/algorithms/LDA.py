import os
import re
from shutil import which
import subprocess
from typing import TYPE_CHECKING, Optional

import dask.array as da
from gensim.models.ldamulticore import LdaMulticore
import numpy as np
import sparse

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

    def fit_transform(self, dataset: "Dataset", name: str) -> TopicModel:
        # https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore
        model = LdaMulticore(
            corpus=dataset.get_gensim_corpus(),
            id2word=dataset.get_gensim_vocab(),
            num_topics=self.n,
            **self.kwargs,
        )
        # To get back DT matrix https://github.com/bmabey/pyLDAvis/blob/master/pyLDAvis/gensim_models.py
        topic_word_matrix = da.from_array(model.get_topics())
        doc_topic_matrix = da.stack(
            [
                da.from_array(
                    sparse.COO(
                        *list(zip(*model.get_document_topics(doc))),
                        shape=self.n,
                    ).todense()
                )
                for doc in dataset.get_gensim_corpus()
            ]
        )

        return TopicModel.from_dask_array(name, topic_word_matrix, doc_topic_matrix)


# https://github.com/RaRe-Technologies/gensim/blob/release-3.8.3/gensim/models/wrappers/ldamallet.py
# https://programminghistorian.org/en/lessons/topic-modeling-and-mallet
# https://stackoverflow.com/questions/42089942/how-to-get-probability-of-words-of-topics-in-mallet
class LDAGS(TMAlgorithm):
    def __init__(
        self,
        n: int,
        optimize_interval: int = 20,
        memory: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.n = n
        self.optimize_interval = optimize_interval
        self.memory = memory
        self.kwargs = kwargs

    def fit_transform(self, dataset: "Dataset", name: str) -> TopicModel:

        command = (
            "mallet train-topics "
            + f"--input {str(dataset.get_mallet_corpus())} "
            + f"--num-topics {self.n} "
            + f"--optimize-interval {self.optimize_interval} "
            + f"--output-doc-topics {str(common.PROJDIR / (name + '.doc_topic.mallet.out'))} "
            + f"--topic-word-weights-file {str(common.PROJDIR / (name + '.topic_word.mallet.out'))} "
        )

        if get_seed():
            command += f"--random-seed {get_seed()} "
        if os.cpu_count() is not None:
            command += f"--num-threads {os.cpu_count()} "
        for key, value in self.kwargs.items():
            command += f"--{key} {value} "

        if self.memory:
            mallet_path = which("mallet")
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

        subprocess.run(command).check_returncode()

        # topic_word_matrix
        # doc_topic_matrix

        return TopicModel.from_dask_array(name, topic_word_matrix, doc_topic_matrix)
