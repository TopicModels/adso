import os
import subprocess

from gensim.models.ldamulticore import LdaMulticore

from .. import common
from ..common import get_seed
from ..data import Dataset
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm


class LDA_VB(TMAlgorithm):
    def __init__(self, n: int, **kwargs) -> None:
        self.n = n
        self.kwargs = kwargs

    def fit_transform(self, dataset: Dataset, name: str) -> TopicModel:
        # https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore
        model = LdaMulticore(
            corpus=dataset.get_gensim_corpus(),
            id2word=dataset.get_gensim_vocab(),
            num_topic=self.n,
            **self.kwargs,
        )
        # To get back DT matrix https://github.com/bmabey/pyLDAvis/blob/master/pyLDAvis/gensim_models.py
        topic_word_matrix = model.get_topics()
        doc_topic_matrix = model.state.get_lambda()

        return TopicModel.from_dask_array(name, topic_word_matrix, doc_topic_matrix)


# https://github.com/RaRe-Technologies/gensim/blob/release-3.8.3/gensim/models/wrappers/ldamallet.py
# https://programminghistorian.org/en/lessons/topic-modeling-and-mallet
# https://stackoverflow.com/questions/42089942/how-to-get-probability-of-words-of-topics-in-mallet
class LDA_GS(TMAlgorithm):
    def __init__(self, n: int, optimize_interval: int = 20, **kwargs) -> None:
        self.n = n
        self.optimize_interval = optimize_interval
        self.kwargs = kwargs

    def fit_transform(self, dataset: Dataset, name: str) -> TopicModel:

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

        subprocess.run(command).check_returncode()

        # topic_word_matrix
        # doc_topic_matrix

        return TopicModel.from_dask_array(name, topic_word_matrix, doc_topic_matrix)
