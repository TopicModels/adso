from typing import Iterable

import nltk

from .common import nltk_download


def nltk_stopwords(lang: str = "english") -> Iterable[str]:
    nltk_download("stopwords")
    return nltk.corpus.stopwords.words(lang)


# http://mallet.cs.umass.edu/import-stoplist.php

# https://github.com/amarallab/stopwords
