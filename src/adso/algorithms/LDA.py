from gensim.models.ldamulticore import LdaMulticore

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
            **self.kwargs
        )
        # To get back DT matrix https://github.com/bmabey/pyLDAvis/blob/master/pyLDAvis/gensim_models.py
        topic_word_matrix = model.get_topics()
        doc_topic_matrix = model.state.get_lambda()

        return TopicModel.from_dask_array(name, topic_word_matrix, doc_topic_matrix)


# Old Gensim MALLET wrapper
# https://radimrehurek.com/gensim_3.8.3/models/wrappers/ldamallet.html
# https://github.com/RaRe-Technologies/gensim/blob/release-3.8.3/gensim/models/wrappers/ldamallet.py
# https://docs.python.org/3/library/subprocess.html
# http://mallet.cs.umass.edu/classification.php
# https://programminghistorian.org/en/lessons/topic-modeling-and-mallet
