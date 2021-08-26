from __future__ import annotations

import zipfile
from functools import reduce
from math import ceil, floor
from pathlib import Path
from typing import Callable, Optional

import dask.array as da
import pandas as pd
import requests
import sparse
from more_itertools import chunked

from .. import common
from ..common import compute_hash
from ..data import LabeledDataset


def get_gutenberg(
    name: str,
    overwrite: bool = False,
    min_book_per_shelf: int = 10,
    min_count: int = 3,
    max_count: Optional[int] = None,
    min_freq: float = 0.0,
    max_freq: float = 1.0,
    filter: Optional[Callable[[str], bool]] = None,
    chunksize: int = 100,
    **kwargs
) -> LabeledDataset:
    # https://github.com/pgcorpus/
    GUTENDIR = common.DATADIR / "gutenberg"
    GUTENDIR.mkdir(exist_ok=True, parents=True)
    countpath = GUTENDIR / "SPGC-counts-2018-07-18.zip"
    metapath = GUTENDIR / "SPGC-metadata-2018-07-18.csv"
    shelfpath = GUTENDIR / "gutenberg-analysis.zip"
    counthash = "bccfbdf00caa906d84344cf335cc96ee"
    metahash = "a2d5f325f13846cbec2fd21d982b4ef4"
    shelfhash = "a02b0108c47da4578de497e552df55f4"

    def download(url: str, path: Path) -> None:
        path.open("wb").write(
            requests.get(
                url,
                allow_redirects=True,
            ).content
        )

    def get(url: str, path: Path, hash: str) -> None:
        if path.exists():
            if compute_hash(path) == hash:
                pass
            else:
                download(url, path)
        else:
            download(url, path)
        assert compute_hash(path) == hash

    get(
        "https://zenodo.org/record/2422561/files/SPGC-counts-2018-07-18.zip?download=1",
        countpath,
        counthash,
    )
    get(
        "https://zenodo.org/record/2422561/files/SPGC-metadata-2018-07-18.csv?download=1",
        metapath,
        metahash,
    )

    get(
        "https://github.com/pgcorpus/gutenberg-analysis/archive/refs/heads/master.zip",
        shelfpath,
        shelfhash,
    )

    metadata = pd.read_csv(metapath)
    metadata = metadata[metadata.language.notna()]
    metadata = metadata[metadata.language == "['en']"]

    def clean(s: str) -> str:
        suffix = "_(Bookshelf)"
        if s.endswith(suffix):
            return s[: -len(suffix)]
        return s

    with zipfile.ZipFile(shelfpath) as z:
        bookshelves = pd.read_pickle(
            z.open("gutenberg-analysis-master/data/bookshelves_raw.p")
        )
    bookshelves = bookshelves.fillna(False)
    book_for_shelf = bookshelves.sum(axis=0)
    book_for_shelf = book_for_shelf[book_for_shelf >= min_book_per_shelf]
    bookshelves = bookshelves[
        [c for c in bookshelves.columns if c in book_for_shelf.index.tolist()]
    ]
    bookshelves = bookshelves[bookshelves.sum(axis=1) == 1]
    book_for_shelf = bookshelves.sum(axis=0)
    book_for_shelf = book_for_shelf[book_for_shelf >= min_book_per_shelf]
    bookshelves = bookshelves[
        [c for c in bookshelves.columns if c in book_for_shelf.index.tolist()]
    ]
    bookshelves = bookshelves.reset_index().melt(id_vars="index", var_name="Bookshelf")
    bookshelves = bookshelves[bookshelves.value == True][["index", "Bookshelf"]]
    bookshelves.Bookshelf = bookshelves.Bookshelf.apply(clean)

    data = metadata.merge(bookshelves, how="inner", left_on="id", right_on="index")

    del metadata
    del bookshelves

    data = data[["id", "Bookshelf"]]
    data = data[data.id != "PG8700"]  # empty file
    with zipfile.ZipFile(countpath) as z:
        files = z.namelist()
    data["path"] = "SPGC-counts-2018-07-18/" + data.id + "_counts.txt"
    data = data[data.path.isin(files)]

    def get_data(data: pd.Dataframe) -> pd.Dataframe:
        with zipfile.ZipFile(countpath) as z:
            data["text"] = data.apply(
                lambda row: [
                    s.strip().decode("utf-8").split("\t")
                    for s in z.open(row.path, "r").readlines()
                ],
                axis=1,
            )
        data = data.explode("text")
        data[["word", "count"]] = data["text"].tolist()
        data.drop(columns=["text"])
        data["count"] = data["count"].astype(int)
        if filter is not None:
            data = data[data["word"].map(filter)]
        return data

    data = reduce(
        pd.concat,
        map(
            get_data,
            [data[data.id.isin(ids)] for ids in chunked(data.id.unique(), chunksize)],
        ),
    )

    if min_freq > 0 or max_freq < 1:
        total_count = data["count"].sum()

    if min_freq > 0:
        min_count = max(min_count, ceil(min_freq * total_count))

    if max_freq < 1:
        if max_count is not None:
            max_count = min(max_count, floor(max_freq * total_count))
        else:
            max_count = floor(max_freq * total_count)

    word_count = data.groupby("word").sum()[["word", "count"]]

    min_word = word_count[word_count.count >= min_count]["word"].unique().tolist()
    if max_count is None:
        vocab = min_word
    else:
        max_word = word_count[word_count.count <= max_count]["word"].unique().tolist()
        vocab = list(set(min_word) | set(max_word))
        del max_word
    del min_word
    del word_count

    data = data[data.word.isin(vocab)]

    doc_count = data.groupby("word").sum()[["id", "count"]]
    docs = doc_count[doc_count.count > 0]["id"].unique().tolist()
    data = data[data.id.isin(docs)]
    del doc_count

    data["word"] = data["word"].map(lambda word: vocab.index(word))
    data["id"] = data["id"].map(lambda id: docs.index(id))

    return LabeledDataset.from_count_matrix(
        name,
        da.array(
            sparse.COO(
                data[["id", "word"]].to_numpy().T,
                data=data.count.to_numpy(),
                fill_value=0,
            )
        ),
        da.array(vocab),
        da.array(data.Bookshelf.to_numpy()),
        overwrite=overwrite,
    )
