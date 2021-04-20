"""Common variable and function for data submodule."""

import hashlib
from pathlib import Path
from typing import Iterable

import nltk

from .. import common


def compute_hash(path: Path) -> str:
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


def nltk_download(id: str) -> None:
    """Wrapper for :func:`nltk.downloader.download`, to download nltk data inside
    adso directory.

    Args:
        id (str): id of the required downloadable data. Cfr
            <http://www.nltk.org/nltk_data/>
    """
    NLTKDIR = common.DATADIR / "nltk"
    NLTKDIR.mkdir(exist_ok=True, parents=True)
    if NLTKDIR not in nltk.data.path:
        nltk.data.path.append(NLTKDIR)
    nltk.downloader.download(id, download_dir=NLTKDIR)


def tokenize_and_stem(doc: str) -> Iterable[str]:
    nltk_download("punkt")
    tokenizer = nltk.tokenize.word_tokenize
    stemmer = nltk.stem.SnowballStemmer("english")
    return map(stemmer.stem, tokenizer(doc))


def get_nltk_stopwords() -> Iterable[str]:
    nltk_download("stopwords")
    return nltk.corpus.stopwords.words("english")
