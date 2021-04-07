"""Common variable and function for data submodule."""

import abc

import nltk

from ..common import ADSODIR

NLTKDIR = ADSODIR / "NLTK"


def nltk_download(id: str):
    """Wrapper for :func:`nltk.downloader.download`, to download nltk data inside
    adso directory.

    Args:
        id (str): id of the required downloadable data. Cfr
            <http://www.nltk.org/nltk_data/>
    """
    return nltk.downloader.download(id, download_dir=NLTKDIR)


class Transformer(abc.ABC):
    """Abstract class for data manipulation class."""

    @abc.abstractmethod
    def fit(data):
        """Find the parameters necessary to trasform the data.

        Args:
            data ([type]): input data
        """
        pass

    @abc.abstractmethod
    def transform(data):
        """Transform the data.

        Args:
            data ([type]): input data
        """
        pass

    @abc.abstractmethod
    def fit_transform(data):
        """Combine :func:`~topicmodel.common.TopicModel.fit` and
        :func:`~topicmodel.common.TopicModel.transform`.

        This is to be preferred to calling fit and transform subsequently.

        Args:
            data ([type]): input data
        """
        pass
