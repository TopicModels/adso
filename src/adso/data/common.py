"""Common variable and function for data submodule."""

import hashlib

from pathlib import Path

from ..common import ADSODIR

DATADIR = ADSODIR / "data"


def hash(path: Path) -> str:
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
