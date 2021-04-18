"""Common variable and function for data submodule."""

import hashlib
from pathlib import Path

from .. import common as adso_common

DATADIR = adso_common.ADSODIR / "data"
DATADIR.mkdir(exist_ok=True, parents=True)


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
