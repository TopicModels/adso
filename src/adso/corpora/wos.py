from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import requests

from .. import common
from ..common import compute_hash
from ..data import LabeledDataset


def get_wos46985(
    name: str, overwrite: bool = False, subfields=False, **kwargs
) -> LabeledDataset:
    # https://data.mendeley.com/datasets/9rw3vkcfy4/6
    WOSDIR = common.DATADIR / "wos"
    WOSDIR.mkdir(exist_ok=True, parents=True)
    wos_path = WOSDIR / "WebOfScience.zip"
    wos_hash = "5cd3753b77deda2c3bde8a7511041fee"

    def download() -> None:
        wos_path.open("wb").write(
            requests.get(
                "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded",
                allow_redirects=True,
            ).content
        )

    if wos_path.exists():
        if compute_hash(wos_path) == wos_hash:
            pass
        else:
            download()
    else:
        download()

    assert compute_hash(wos_path) == wos_hash

    with zipfile.ZipFile(wos_path) as z:
        with z.open("Meta-data/Data.xlsx") as f:
            df = pd.read_excel(f)

    if subfields:
        col = "area"
    else:
        col = "Domain"

    labels = df[col].to_numpy(dtype=np.dtype(str))
    data = df["Abstract"].to_numpy(dtype=np.dtype(str))

    return LabeledDataset.from_array(
        name,
        labels,
        data,
        overwrite=overwrite,
    )
