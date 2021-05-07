from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple

from ..common import Data

if TYPE_CHECKING:
    from ..data import Dataset


class Algorithm(Data, ABC):
    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, dataset: "Dataset") -> Any:
        raise NotImplementedError


class TMAlgorithm(ABC):
    @abstractmethod
    def fit_transform(self, dataset: "Dataset", name: str) -> Tuple[Any, ...]:
        raise NotImplementedError
