import pickle
from itertools import chain
from pathlib import Path
from typing import Callable, Iterable, Optional

from dask_ml.feature_extraction.text import CountVectorizer

from ..common import compute_hash
from ..data.common import nltk_download, tokenize_and_stem
from .common import Algorithm


class Vectorizer(Algorithm):
    def __init__(
        self,
        path: Path,
        overwrite: bool = False,
        tokenizer: Optional[Callable] = tokenize_and_stem,
        stop_words: Optional[Iterable[str]] = None,
        strip_accents: Optional[str] = "unicode",
        **kwargs
    ) -> None:
        self.path = path
        if tokenizer == tokenize_and_stem:
            nltk_download("punkt")
        if (stop_words is not None) and (tokenizer is not None):
            stop_words = set(chain.from_iterable([tokenizer(sw) for sw in stop_words]))
        model = CountVectorizer(
            tokenizer=tokenizer, stop_words=stop_words, strip_accents=strip_accents
        )
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            self.save(model)

    def save(self, model: CountVectorizer) -> None:  # type: ignore[override]
        pickle.dump(
            model,
            self.path.open("wb"),
        )
        self.update_hash()

    def get(self) -> CountVectorizer:
        if self.hash == compute_hash(self.path):
            return pickle.load(self.path.open("rb"))
        else:
            raise RuntimeError("Different hash")
