"""Common variable and function for adso module."""

import os
import random
from pathlib import Path

from .data import common as data_common

import numpy as np

ENV_ADSODIR = os.getenv("ADSODIR")
if ENV_ADSODIR is not None:
    ADSODIR = Path(ENV_ADSODIR)
else:
    ADSODIR = Path.home() / ".adso"
del ENV_ADSODIR

PROJDIR = ADSODIR / "default"

ADSODIR.mkdir(exist_ok=True, parents=True)
PROJDIR.mkdir(exist_ok=True, parents=True)


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
    ADSODIR = Path(path)
    ADSODIR.mkdir(exist_ok=True, parents=True)
    data_common.DATADIR = ADSODIR / "data"
    data_common.DATADIR.mkdir(exist_ok=True, parents=True)


def set_project_name(name: str) -> None:
    global PROJDIR
    PROJDIR = ADSODIR / name
    PROJDIR.mkdir(exist_ok=True, parents=True)
