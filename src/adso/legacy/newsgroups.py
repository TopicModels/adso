"""20newgroups datatset.

Download the 20newsgroup dataset from `Jason Rennie's site
    <http://qwone.com/~jason/20Newsgroups/>` and import it as :class:`LabelledDataset`
"""

import tarfile
from typing import Tuple, Union

import requests

from .common import DATADIR, load_txt
from .dataset import LabelledDataset


def load_20newsgroups(
    split: str = "all",
) -> Union[LabelledDataset, Tuple[LabelledDataset, LabelledDataset]]:
    """Load the 20newsgroups dataset with labels from
    <http://qwone.com/~jason/20Newsgroups/>`

    Args:
        split (str, optional): which part of the dataset have to be load:

            * "all" for a single :class:`LabelledDataset` with all the available
              documents
            * "train" for the train split only (deterministic)
            * "test" for the test split only (deterministic)
            * "both" for the tuple (train, test)

            Defaults to "all".

    Returns:
        Union[LabelledDataset, Tuple[LabelledDataset, LabelledDataset]]:
            the 20newsgroups dataset with labels, as LabelledDataset
    """
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
