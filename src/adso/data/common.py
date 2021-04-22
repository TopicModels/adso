"""Common variable and function for data submodule."""

from typing import Iterable

import nltk

from .. import common


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
    tokenizer = nltk.tokenize.word_tokenize
    stemmer = nltk.stem.SnowballStemmer("english")
    return map(stemmer.stem, tokenizer(doc))


def get_nltk_stopwords() -> Iterable[str]:
    nltk_download("stopwords")
    return nltk.corpus.stopwords.words("english")
