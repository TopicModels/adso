from __future__ import annotations

import json
import zipfile
from itertools import chain, repeat

import requests

from .. import common
from ..common import compute_hash
from ..data import LabeledDataset


def get_salesforce(
    name: str, overwrite: bool = False, latin: bool = True, **kwargs
) -> LabeledDataset:
    # https://github.com/salesforce/localization-xml-mt
    SALESDIR = common.DATADIR / "salesforce"
    SALESDIR.mkdir(exist_ok=True, parents=True)
    sales_path = SALESDIR / "salesforce_repo.zip"
    sales_hash = "8dd077148fd20ad3511d49c658214832"

    def download() -> None:
        sales_path.open("wb").write(
            requests.get(
                "https://github.com/salesforce/localization-xml-mt/archive/refs/heads/master.zip",
                allow_redirects=True,
            ).content
        )

    if sales_path.exists():
        if compute_hash(sales_path) == sales_hash:
            pass
        else:
            download()
    else:
        download()

    assert compute_hash(sales_path) == sales_hash

    def filterpath(path: str, latin: bool = True) -> bool:
        if not path.endswith(".json"):
            return False
        if "scripts" in path:
            return False
        if latin:
            if "ru" in path:
                return False
            if "zh" in path:
                return False
            if "ja" in path:
                return False
        return True

    with zipfile.ZipFile(sales_path) as z:
        jsons = [
            json.loads(z.open(path).read()) for path in z.namelist() if filterpath(path)
        ]
    values = [(f["lang"], f["text"].values()) for f in jsons]
    data = filter(
        lambda t: len(t[1].split()) >= 5,
        chain(*[tuple(zip(repeat(t[0]), t[1])) for t in values]),
    )

    return LabeledDataset.from_iterator(
        name,
        data,
        overwrite=overwrite,
    )
