import nltk

from ..common import ADSODIR

NLTKDIR = ADSODIR / "NLTK"


def nltk_download(id: str):
    return nltk.downloader.download(id, download_dir=NLTKDIR)
