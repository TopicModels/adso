import requests
import tarfile

from typing import Tuple, Union

from .common import DATADIR, load_txt
from .dataset import LabelledDataset


def load_20newsgroups(
    split="all",
) -> Union[LabelledDataset, Tuple[LabelledDataset, LabelledDataset]]:
    newsdir = DATADIR / "20newsgroups"
    newsdir.mkdir(parents=True, exist_ok=True)
    if not any(newsdir.iterdir()):
        data = requests.get(
            "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz",
            allow_redirects=True,
            stream=True,
        )
        with (newsdir / "20news.tar.gz").open("wb") as file:
            file.write(data.content)
        tarfile.open(name=(newsdir / "20news.tar.gz"), mode="r|gz").extractall(
            path=newsdir
        )

    train = load_txt(newsdir / "20news-bydate-test", label=True, extension=None)
    test = load_txt(newsdir / "20news-bydate-test", label=True, extension=None)

    if split == "both":
        return train, test
    elif split == "train":
        return train
    elif split == "test":
        return test
    elif split == "all":
        return train + test
    else:
        raise ValueError("Split can be both, train, test or all")
