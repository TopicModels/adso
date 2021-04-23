from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..common import Data
from ..data.topicmodel import TopicModel

if TYPE_CHECKING:
    from ..data import Dataset


class Algorithm(Data, ABC):
    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, dataset: "Dataset") -> Any:
        raise NotImplementedError


class TMAlgorithm(Algorithm, ABC):
    def __init__(self, path: Path, name: str) -> None:
        super().__init__(path)
        self.name = name

    @abstractmethod
    def fit_transform(self, dataset: "Dataset", path: Path) -> TopicModel:  # type: ignore[override]
        raise NotImplementedError
