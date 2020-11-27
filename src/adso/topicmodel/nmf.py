"""Non Negative Matrix Factorization.

AHCLS algorithm from Langville et al. ArXiv:1407.7299
"""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy as sp

from .common import TopicModel


class NMF(TopicModel):
    def __init__(
        self: NMF,
        n_topic: int = 10,
        max_iter: int = 200,
        tolerance: Union[float, None] = None,
        lambdaW: float = 0.5,
        lambdaH: float = 0.5,
        alphaW: float = 0.5,
        alphaH: float = 0.5,
        method: str = "ACLS",
    ) -> None:
        if n_topic > 0:
            self.n_topic = n_topic
        else:
            raise ValueError("n_topic must be positive")
        if max_iter > 0:
            self.max_iter = max_iter
        else:
            raise ValueError("max_iter must be positive")
        if lambdaW >= 0 and lambdaH >= 0:
            self.lambdaW = lambdaW
            self.lambdaH = lambdaH
        else:
            raise ValueError("lambda parameters must be non negative")
        if (1 >= alphaW >= 0) and (1 >= alphaH >= 0):
            self.alphaW = alphaW
            self.alphaH = alphaH
        else:
            raise ValueError("alpha parameters must be in [0,1]")
        if tolerance is None or tolerance > 0:
            self.tolerance = tolerance
        else:
            raise ValueError("tolerance must be positive")
        if method in ["ALS", "ACLS", "AHCLS"]:
            self.method = method
        else:
            raise ValueError("Only ALS, ACLS, AHCLS methods are implemented")

    def fit(self: NMF, data: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
        self.fit_transform(data)
        return self.H

    def transform(self: NMF, data: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
        # solve Ht @ Wt = At equivalent to W @ H = A
        if data.shape[1] != self.H.shape[1]:
            raise ValueError(
                "shape of data is incompatible with the shape of the previuosly fitted topic-term matrix. The same vocab used for fitted data must be used for newer data"
            )
        return sp.sparse.linalg.spsolve(
            self.H.transpose(), data.transpose()
        ).transpose()

    def fit_transform(
        self: NMF, data: sp.sparse.spmatrix
    ) -> tuple(sp.sparse.spmatrix, sp.sparse.spmatrix, int):
        A = data
        n_doc, n_term = data.shape
        n_topic = self.n_topic
        tolerance = self.tolerance

        if self.method == "AHCLS":
            betaH = ((1 - self.alphaH) * (n_topic ** 0.5) + self.alphaH) ** 2
            betaW = ((1 - self.alphaW) * (n_topic ** 0.5) + self.alphaW) ** 2

        if n_term / n_topic > 20:
            p = 20
        else:
            p = max(int(n_term / n_topic), 1)

        W = sp.sparse.csr_matrix(
            np.mean(
                np.resize(
                    data[
                        :, np.random.choice(n_term, size=p * n_topic, replace=False)
                    ].todense(),
                    (n_doc, n_topic, p),
                ),
                axis=2,
            )
        )

        if tolerance is not None:
            trAtA = (A.transpose() @ A).diagonal().sum()

            def error(
                H: sp.sparse.spmatrix, WtA: sp.sparse.spmatrix, WtW: sp.sparse.spmatrix
            ) -> float:
                return np.asscalar(
                    trAtA
                    - 2 * (H.transpose() @ WtA).diagonal().sum()
                    + (H.transpose() @ (WtW @ H)).diagonal().sum()
                )

        if self.method == "AHCLS":

            def updateH(WtW: sp.sparse.matrix) -> sp.sparse.matrix:
                return sp.sparse.linalg.spsolve(
                    WtW.todense()
                    + self.lambdaH * betaH * sp.sparse.eye(n_topic)
                    - self.lambdaH,
                    WtA,
                )

            def updateW(H: sp.sparse.matrix) -> sp.sparse.matrix:
                return sp.sparse.linalg.spsolve(
                    (H @ H.transpose()).todense()
                    + self.lambdaW * betaW * sp.sparse.eye(n_topic)
                    - self.lambdaW,
                    H @ A.transpose(),
                ).transpose()

        elif self.method == "ACLS":

            def updateH(WtW: sp.sparse.matrix) -> sp.sparse.matrix:
                return sp.sparse.linalg.spsolve(
                    WtW - self.lambdaH * sp.sparse.eye(n_topic),
                    WtA,
                )

            def updateW(H: sp.sparse.matrix) -> sp.sparse.matrix:
                return sp.sparse.linalg.spsolve(
                    H @ H.transpose() - self.lambdaW * sp.sparse.eye(n_topic),
                    H @ A.transpose(),
                ).transpose()

        elif self.method == "ALS":

            def updateH(WtW: sp.sparse.matrix) -> sp.sparse.matrix:
                return sp.sparse.linalg.spsolve(
                    WtW,
                    WtA,
                )

            def updateW(H: sp.sparse.matrix) -> sp.sparse.matrix:
                return sp.sparse.linalg.spsolve(
                    H @ H.transpose(),
                    H @ A.transpose(),
                ).transpose()

        for iter in range(self.max_iter):
            WtW = W.transpose() @ W
            WtA = W.transpose() @ A
            H = updateH(WtW)
            H[H < 0] = 0
            W = updateW(H)
            W[W < 0] = 0

            if iter % 10 == 0:
                if tolerance is not None:
                    if tolerance > error(H, WtA, WtW):
                        self.H = H
                        return W, H, iter + 1

        self.H = H
        return W, H, iter + 1
