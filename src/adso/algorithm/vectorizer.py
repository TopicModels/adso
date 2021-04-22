import pickle
from itertools import chain
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from dask_ml.feature_extraction.text import CountVectorizer

from ..common import Data
from ..data.common import nltk_download, tokenize_and_stem
from .common import Algorithm


class Vectorizer(Data, Algorithm):
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
        self.model = CountVectorizer(
            tokenizer=tokenizer, stop_words=stop_words, strip_accents=strip_accents
        )
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            self.save()

    def save(self) -> None:
        pickle.dump(
            self.model,
            self.path.open("wb"),
        )
        self.update_hash()

    @classmethod
    def load(cls, path: Union[Path, str], hash: Optional[str]) -> "Vectorizer":  # type: ignore[override]
        return super().load(path, hash)  # type: ignore[return-value]

    def get(self) -> CountVectorizer:
        return self.model
