from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import subprocess
from shutil import which

import dask.array as da
import dask.dataframe as dd
import numpy as np
import sparse

from .. import common
from ..common import get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data import Dataset


class TopicMapping(TMAlgorithm):
    # https://amaral.northwestern.edu/resources/software/topic-mapping
    # https://bitbucket.org/andrealanci/topicmapping/src/master/ReadMe.pdf
    def __init__(self, lag: int = 1000, **kwargs) -> None:
        self.lag = lag
        self.kwargs = kwargs

    def fit_transform(
        self, dataset: Dataset, name: str
    ) -> Tuple[TopicModel, Tuple[int]]:

        (common.PROJDIR / "topicmapping").mkdir(exist_ok=True, parents=True)
        out_path = common.PROJDIR / "topicmapping" / name

        command = (
            "topicmap "
            + f"-f {str(dataset.get_topicmapping_corpus())} "
            + f"-o {str(out_path)} "
            + f"-lag {str(self.lag)} "
        )

        if get_seed():
            command += f"-seed {get_seed()} "

        for key, value in self.kwargs.items():
            command += f"-{key} {value} "

        print(command)

        if which("topicmap"):
            subprocess.run(command, shell=True, check=True)
        else:
            raise RuntimeError(
                "topicmap not found, check to be available in PATH (topicmap exec and not bin folder). Install it with conda (topicmapping package) should resolve the issue"
            )

        doc_topic_matrix = da.from_array(
            np.genfromtxt(out_path / "lda_gammas_final.txt")
        )

        n_topic = doc_topic_matrix.shape[1]

        vocab = dd.read_csv(
            out_path / "word_wn_count.txt",
            sep=" ",
            header=None,
            usecols=[0, 1],
        ).rename(columns={0: "adso_id", 1: "tm_id"})

        with (out_path / "lda_betas_sparse_final.txt").open() as inf:
            with (out_path / "lda_betas_sparse_final_tab.txt").open("x") as outf:
                for row in inf:
                    item = row.strip().split(" ")
                    if len(item) % 2 != 1:
                        raise RuntimeError("Unable to parse topicmap output")
                    col = item[0]
                    for i in range((len(item) - 1) // 2):
                        outf.write(
                            " ".join([col, item[2 * i + 1], item[2 * i + 2]]) + "\n"
                        )

        topic_word_df = dd.read_csv(
            out_path / "lda_betas_sparse_final_tab.txt",
            sep=" ",
            header=None,
        ).rename(columns={0: "doc", 1: "tm_id", 2: "val"})

        topic_word_df = topic_word_df.merge(vocab, how="left", on="tm_id")

        topic_word_matrix = da.from_array(
            sparse.COO(
                topic_word_df[["doc", "adso_id"]].values.T,
                data=topic_word_df["val"].values,
                fill_value=0,
                shape=(n_topic, dataset.n_word()),
            ).todense()
        )

        return (
            TopicModel.from_dask_array(
                name,
                topic_word_matrix,
                doc_topic_matrix,
            ),
            (n_topic,),
        )
