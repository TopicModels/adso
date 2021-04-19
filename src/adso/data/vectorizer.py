import pickle

from dask_ml.feature_extraction.text import CountVectorizer

from .corpus import Corpus


class Vectorizer(Corpus):
    def get(self) -> CountVectorizer:
        if self.hash == hash(self.path):
            return pickle.load(self.path.open("rb"))
        else:
            raise RuntimeError
