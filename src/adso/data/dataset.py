"""Dataset class.

Define data-container for other classes.
"""

from abc import ABC
from typing import Any, Optional, Union

import json
import os

from pathlib import Path

from ..common import PROJDIR


class Corpus (ABC):

    def __init__(self: 'Corpus', path: Path) -> None:
        self.path = Path
        self.hash: str
        self.format: str = 'raw'

    def get(self: 'Corpus') -> Any:
        pass

    def to_json(self: 'Corpus') -> dict:
        return {
            'format': self.format,
            'path': self.path,
            'hash': self.hash
        }


class Raw(Corpus):
    pass


class Dataset:
    """Dataset class."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = PROJDIR / self.name
        # maybe add existence check?
        self.path.mkdir(exist_ok=True, parents=True)

        self.raw: Optional[Raw] = None

    def to_json(self) -> dict:
        save = {'name': self.name, 'path': self.path}
        if self.raw is not None:
            save['raw'] = self.raw.to_json()
        return save

    def save(self) -> None:
        with (self.path / (self.name + '.json')).open(mode='w') as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> 'Dataset':
        path = Path(path)
        if path.is_dir():
            with (path / (path.name + '.json')).open(mode='r') as f:
                data = json.load(f)
        else:
            with path.open(mode='r') as f:
                data = json.load(f)
            path = path.parent

        dataset = cls(data['name'])
        dataset.path = path

        dataset.save()
        return dataset


class LabeledDataset(Dataset):

    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.raw_label: Optional[Raw] = None

    def to_json(self) -> dict:
        save = super().to_json()
        if self.raw_label is not None:
            save['raw_label'] = self.raw_label.to_json()
        return save
