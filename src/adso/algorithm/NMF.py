import pickle

import sklean as sk

from ..common import PROJDIR, get_seed, compute_hash
from .common import TMAlgorithm


class NMF(TMAlgorithm):
    def __init__(self, name: str, n: int, overwrite: bool = False, **kwargs) -> None:
        path = PROJDIR / (name + ".pickle")
        super().__init__(path, name)
        model = sk.decomposition.NMF(n_components=n, random_state=get_seed(), **kwargs)
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            self.save(model)

    def save(self, model: sk.decomposition.NMF) -> None:  # type: ignore[override]
        pickle.dump(
            model,
            self.path.open("wb"),
        )
        self.update_hash()

    def get(self) -> sk.decomposition.NMF:
        if self.hash == compute_hash(self.path):
            return pickle.load(self.path.open("rb"))
        else:
            raise RuntimeError("Different hash")
