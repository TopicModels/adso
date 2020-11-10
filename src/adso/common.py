from pathlib import Path
import random

import numpy as np

ADSODIR = Path.home() / ".adso"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility. Take care of python random library, numpy and tensorflow.

    Args:
        seed (int): the value choosen as seed for the random generators
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
