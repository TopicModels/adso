"""Common variable and function for adso module."""

import os
import random
from pathlib import Path

import numpy as np

ADSODIR = os.getenv("ADSODIR")
if ADSODIR is not None:
    ADSODIR = Path(ADSODIR)
else:
    ADSODIR = Path.home() / ".adso"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Take care of python random library and numpy.

    Args:
        seed (int): the value choosen as seed for the random generators
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
