import abc

import nltk

from ..common import ADSODIR

NLTKDIR = ADSODIR / "NLTK"


def nltk_download(id: str):
    return nltk.downloader.download(id, download_dir=NLTKDIR)


class Transformer(abc.ABC):
    @abc.abstractmethod
    def fit(data):
        pass

    @abc.abstractmethod
    def transform(data):
        pass

    @abc.abstractmethod
    def fit_transform(data):
        pass
