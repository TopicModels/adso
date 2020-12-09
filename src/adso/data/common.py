"""Common variable and function for data submodule."""

import os
from itertools import chain
from pathlib import Path
from typing import Union


from .dataset import Dataset, LabelledDataset
from ..common import ADSODIR

DATADIR = ADSODIR / "data"


def load_txt(
    path: Union[str, bytes, os.PathLike],
    lines: bool = False,
    label: bool = False,
    extension: Union[str, None] = "txt",
    encoding: str = "utf-8",
    ignore_errors: bool = True,
) -> Dataset:
    """Load text files as dataset.

    Load a dataset composed by text files.
    This function scan the given directory and all its subdirectories and read all the
    files in them. The single items for the dataset could be either the files or each
    lines in each files. In the first case the folder's name will be used as label,
    in the latter the file's name.

    Args:
        path (Union[str, bytes, os.PathLike]): the directory to be scanned for the
            datatset.
        lines (bool, optional): if each item of the datatset is a different file (False)
            or each line in it (True). Defaults to False.
        label (bool, optional): to load the files as a Dataset (False) or a
            LabelledDataset, i.e. a dataset with labelled data (True). The folder names
            (if lines=False) or the file names (if lines=False) are used as labels.
            Defaults to False.
        extension (Union[str, None], optional): the extension of the file to be read.
            If None read all files. Defaults to "txt".
        encoding (str, optional): encoding for the files, passed to
            pathlib.Path.read_text(). Defaults to "utf-8".
        ignore_errors (bool, optional): whenever ignore unreadable characters in files
            or raise an error. Defaults to True.

    Returns:
        Dataset: return a Dataset, as defined in dataset.py file
    """
    path = Path(path)
    if extension is None:
        extension = ""
    else:
        extension = "." + extension

    if ignore_errors:
        errors = "ignore"
    else:
        errors = None

    if label:
        if lines:
            return LabelledDataset(
                list(
                    chain.from_iterable(
                        map(
                            lambda f: map(
                                lambda line: (line, f.stem),
                                f.read_text(
                                    encoding=encoding, errors=errors
                                ).splitlines(),
                            ),
                            filter(
                                lambda f: f.is_file(), path.glob("**/*" + extension)
                            ),
                        )
                    )
                )
            )
        else:  # label and not lines
            return LabelledDataset(
                list(
                    map(
                        lambda f: (
                            f.read_text(encoding=encoding, errors=errors),
                            f.parent.name,
                        ),
                        filter(lambda f: f.is_file(), path.glob("**/*" + extension)),
                    )
                )
            )
    else:  # not label
        if lines:
            return Dataset(
                list(
                    chain.from_iterable(
                        map(
                            lambda f: f.read_text(
                                encoding=encoding, errors=errors
                            ).splitlines(),
                            filter(
                                lambda f: f.is_file(), path.glob("**/*" + extension)
                            ),
                        )
                    )
                )
            )
        else:  # not label and not lines
            return Dataset(
                list(
                    map(
                        lambda f: f.read_text(encoding=encoding, errors=errors),
                        filter(lambda f: f.is_file(), path.glob("**/*" + extension)),
                    )
                )
            )
