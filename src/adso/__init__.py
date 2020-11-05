"""Adso is a tensorflow-based topic-modelling library."""

from pathlib import Path
import random

import numpy as np
import tensorflow as tf

# Create adso folder
ADSODIR = Path.home() / ".adso"
ADSODIR.mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility. Take care of python random library, numpy and tensorflow.

    Args:
        seed (int): the value choosen as seed for the random generators
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
