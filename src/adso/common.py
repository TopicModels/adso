"""Common variable and function for adso module."""

import os
import random
from pathlib import Path

import numpy as np

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


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Take care of python random library and numpy.

    Args:
        seed (int): the value choosen as seed for the random generators
    """
    random.seed(seed)
    np.random.seed(seed)


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


def setup_dask_client() -> None:
    pass
