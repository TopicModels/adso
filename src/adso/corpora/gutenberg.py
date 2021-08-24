from __future__ import annotations

import zipfile
from itertools import chain
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

from .. import common
from ..common import compute_hash
from ..data import LabeledDataset


def get_gutenberg(
    name: str, overwrite: bool = False, min_book_per_shelf: int = 10, **kwargs
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

    def process(t: Tuple[str, str]) -> Tuple[str, str]:
        with zipfile.ZipFile(countpath) as z:
            return (
                t[0],
                " ".join(
                    list(
                        chain(
                            *[
                                [t[0]] * int(t[1])
                                for t in [
                                    s.decode("utf-8").strip().split("\t")
                                    for s in z.open(
                                        f"SPGC-counts-2018-07-18/{t[1]}_counts.txt", "r"
                                    ).readlines()
                                ]
                            ]
                        )
                    )
                ),
            )

    data = metadata.merge(bookshelves, how="inner", left_on="id", right_on="index")
    data = data[["id", "Bookshelf"]]
    data = data[data.id != "PG8700"]  # empty file
    with zipfile.ZipFile(countpath) as z:
        files = z.namelist()
    data = data[("SPGC-counts-2018-07-18/" + data.id + "_counts.txt").isin(files)]
    iterator = map(
        process, data[["Bookshelf", "id"]].itertuples(name=None, index=False)
    )

    return LabeledDataset.from_iterator(
        name,
        iterator,
        overwrite=overwrite,
    )
