"""Common variable and function for data submodule."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, Union

import dask.array as da
import nltk
import numpy as np
import zarr

from .. import common


def nltk_download(id: str) -> None:
    """Wrapper for :func:`nltk.downloader.download`, to download nltk data inside
    adso directory.

    Args:
        id (str): id of the required downloadable data. Cfr
            <http://www.nltk.org/nltk_data/>
    """
    nltk.downloader.download(id, download_dir=common.NLTKDIR)


def tokenize_and_stem(doc: str) -> Iterable[str]:
    tokenizer = nltk.tokenize.word_tokenize
    stemmer = nltk.stem.SnowballStemmer("english")
    return map(stemmer.stem, tokenizer(doc))


def get_nltk_stopwords() -> Iterable[str]:
    nltk_download("stopwords")
    return nltk.corpus.stopwords.words("english")


def encode(array: da.array) -> da.array:
    if array.dtype.kind == "U":
        itemsize = np.dtype("U1").itemsize
    elif array.dtype.kind == "S":
        itemsize = np.dtype("S1").itemsize
    else:
        raise TypeError("Numpy dtype not recognized")
    return array.map_blocks(
        lambda b: np.char.encode(b, encoding="utf-8"),
        dtype=np.dtype(("S", array.itemsize // itemsize)),
    ).rechunk()


def save_array_to_zarr(array: Union[da.array, Dict[str, da.array]], path: Path) -> None:
    dirpath = path.with_name(path.name + ".dir")
    if isinstance(array, da.Array):
        array.rechunk().to_zarr(
            zarr.open(
                zarr.storage.DirectoryStore(dirpath),
                shape=array.shape,
                dtype=array.dtype,
                chunks=array.chunksize,
                mode="a",
            )
        )
    else:
        group = zarr.open_group(store=zarr.storage.DirectoryStore(dirpath), mode="a")
        for component, array in array.items():
            array.rechunk().to_zarr(
                group.require_dataset(
                    component,
                    array.shape,
                    dtype=array.dtype,
                    chuncks=array.chunksize,
                    overwrite=False,
                )
            )

    if path.suffix == ".zip":
        name = path.with_suffix("")
    else:
        name = path

    shutil.make_archive(str(name), "zip", dirpath)

    if path.suffix != ".zip":
        name.with_suffix(name.suffix + ".zip").rename(path)

    shutil.rmtree(dirpath, ignore_errors=True)
