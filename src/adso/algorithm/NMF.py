import sklean as sk

from ..common import get_seed
from ..data.dataset import Dataset
from .common import TMAlgorithm, TopicModel


class NMF(TMAlgorithm):
    def __init__(self, dataset: Dataset, n: int, **kwargs) -> None:
        self.n = n
        self.model = sk.decomposition.NMF(
            n_components=self.n, random_state=get_seed(), **kwargs
        )
        self.dataset = dataset

    def save(self) -> None:
        pass

    @classmethod
    def load(self) -> "NMF":
        pass

    def get_model(self) -> "TopicModel":
        pass
