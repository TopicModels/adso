"""Trasform string tensor to numerical."""

import nltk

from .common import NLTKDIR, nltk_download
from .tokenizer import Tokenizer
from .vectorizer import FreqVectorizer, TFIDFVectorizer, Vectorizer, Vocab

NLTKDIR.mkdir(exist_ok=True, parents=True)
nltk.data.path.append(NLTKDIR)
