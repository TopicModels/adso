"""Common variable and function for adso module."""

import hashlib
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.special as special
from dask.array.ufunc import wrap_elemwise

ADSODIR = (
    (Path.home() / ".adso")
    if (os.getenv("ADSODIR") is None)
    else Path(os.getenv("ADSODIR"))  # type: ignore[arg-type]
)

PROJDIR = ADSODIR / "default"

ADSODIR.mkdir(exist_ok=True, parents=True)
PROJDIR.mkdir(exist_ok=True, parents=True)

DATADIR = ADSODIR / "data"
DATADIR.mkdir(exist_ok=True, parents=True)

SEED: int = 0

xlogy = wrap_elemwise(special.xlogy, source=special)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Take care of python random library and numpy.

    Args:
        seed (int): the value choosen as seed for the random generators
    """
    global SEED
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)


def get_seed() -> Optional[int]:
    if SEED == 0:
        return None
    else:
        return SEED


def set_adso_dir(path: str) -> None:
    global ADSODIR
    global DATADIR
    ADSODIR = Path(path)
    ADSODIR.mkdir(exist_ok=True, parents=True)
    DATADIR = ADSODIR / "data"
    DATADIR.mkdir(exist_ok=True, parents=True)


def set_project_name(name: str) -> None:
    global PROJDIR
    PROJDIR = ADSODIR / name
    PROJDIR.mkdir(exist_ok=True, parents=True)


def compute_hash(path: Path) -> str:
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()

    with path.open(mode="rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


class Data(ABC):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hash: Union[str, List[str]]
        self.update_hash()

    @abstractmethod
    def get(self) -> Any:
        raise NotImplementedError

    def update_hash(self) -> None:
        self.hash = compute_hash(self.path)

    def serialize(self) -> Dict[str, Union[str, List[str]]]:
        return {
            "format": type(self).__name__,
            "path": str(self.path),
            "hash": self.hash,
        }

    @classmethod
    def load(cls, path: Union[Path, str], hash: Optional[str]) -> "Data":
        path = Path(path)
        if path.is_file():
            corpus = cls(path)
            if (corpus.hash == hash) or (hash is None):
                return corpus
            else:
                raise RuntimeError("Different hash")
        else:
            raise RuntimeError("File doesn't exists")
