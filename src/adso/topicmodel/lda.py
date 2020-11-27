"""Latent Dirichlet Allocation.

From Blei et al. 2003.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy as sp
import sparse

from .common import TopicModel


class LDA(TopicModel):
    def __init__(
        self: LDA,
        n_topic: int = 10,
        max_iter: int = 200,
        tolerance: float = 0,
        epsilon: float = 1e-50,
    ) -> None:
        if n_topic > 0:
            self.n_topic = n_topic
        else:
            raise ValueError("n_topic must be positive")
        if max_iter > 0:
            self.max_iter = max_iter
        else:
            raise ValueError("max_iter must be positive")
        if tolerance >= 0:
            self.tolerance = tolerance
        else:
            raise ValueError("tolerance must be non-negative")
        if 1e-10 > epsilon > 0:
            self.epsilon = epsilon
        else:
            raise ValueError("epsilon must be positive and small")

    # data count documwnt-term matrix
    def fit(
        self: LDA, data: sp.sparse.spmatrix
    ) -> Tuple[np.array, np.array, np.array, np.array, float, int]:
        n_topic = self.n_topic
        n_doc, n_term = data.shape

        data = sparse.COO.from_scipy_sparse(data)

        if n_doc / n_topic > 5:
            p = 5
        else:
            p = max(int(n_doc / n_topic), 1)

        alpha = np.ones(n_topic)
        beta = (
            data[np.random.choice(n_doc, size=p * n_topic, replace=False), :]
            .reshape((n_topic, n_term, p))
            .sum(axis=2)
        )
        beta = beta / beta.sum(axis=1, keepdims=True).todense()

        phi = np.full((n_doc, n_term, n_topic), 1 / n_topic)
        gamma = (
            alpha[np.newaxis, :] + data.sum(axis=1, keepdims=True).todense() / n_topic
        )

        loglikelihood = self._loglikelihood(data, alpha, beta, gamma, phi)
        iter = 0
        error = 1

        while iter < self.max_iter and error > self.tolerance:

            loglikelihood_old = loglikelihood

            gamma, phi = self._estep(alpha, beta, gamma)
            alpha, beta, _ = self._mstep(data, alpha, gamma, phi, n_doc)

            loglikelihood = self._loglikelihood(data, alpha, beta, gamma, phi)

            error = abs((loglikelihood_old - loglikelihood) / loglikelihood_old)
            iter += 1

        beta = beta.tocsr()

        self.alpha = alpha
        self.beta = beta

        return alpha, beta, gamma, phi, loglikelihood, iter

    def transform(self: LDA, data: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
        return np.exp(data @ np.log(self.beta).T).tocsr()

    def fit_transform(self: LDA, data: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
        alpha, beta, gamma, phi, loglikelihood, iter = self.fit(data)
        return self.transform(data), alpha, beta, gamma, phi, loglikelihood, iter

    def _loglikelihood(
        self: LDA,
        data: sp.sparse.matrix,  # dw
        alpha: np.array,  # i
        beta: np.array,  # iw
        gamma: np.array,  # di
        phi: np.array,  # dwi
    ) -> float:
        n_doc, n_term = data.shape
        sum_digamma = sp.special.digamma(gamma) - sp.special.digamma(
            gamma.sum(axis=1, keepdims=True)
        )  # di
        return (
            sp.special.loggamma(alpha.sum()) * n_doc
            - sp.special.loggamma(alpha).sum() * n_doc
            + ((alpha - 1) * (sum_digamma.sum(axis=0))).sum()
            + (phi.sum(axis=1) * sum_digamma).sum()
            + (
                phi
                * sparse.elemwise(np.nan_to_num, np.log(beta + self.epsilon))[
                    :, :, np.newaxis
                ].T
                * data[:, :, np.newaxis]
            ).sum()
            - sp.special.loggamma(gamma.sum(axis=1)).sum()
            + sp.special.loggamma(gamma).sum()
            - ((gamma - 1) * sum_digamma).sum()
            - sparse.elemwise(np.nan_to_num, phi * np.log(phi)).sum()
        )

    def _estep(
        self: LDA,
        alpha: np.array,
        beta: np.array,
        gamma: np.array,
    ) -> Tuple[np.array, np.array]:

        phi = (
            np.exp(sp.special.digamma(gamma))[:, np.newaxis, :]
            * beta[:, :, np.newaxis].T
        )
        phi = sparse.elemwise(np.nan_to_num, phi / phi.sum(axis=2, keepdims=True))
        gamma = alpha[np.newaxis, :] + phi.sum(axis=1, keepdims=False).todense()

        return gamma, phi

    def _mstep(
        self: LDA,
        data: sp.sparse.matrix,
        alpha: np.array,
        gamma: np.array,
        phi: np.array,
        n_doc: int,
    ) -> Tuple[np.array, np.array, float]:

        beta = (phi * data[:, :, np.newaxis]).sum(axis=0).T
        beta = beta / beta.sum(axis=1, keepdims=True).todense()
        alpha, iter_alpha = self._update_alpha(alpha, gamma, n_doc)

        return alpha, beta, iter_alpha

    def _update_alpha(
        self: LDA, alpha: np.array, gamma: np.array, n_doc: int
    ) -> Tuple[np.array, int]:
        iter = 0
        error = 1
        init_alpha = alpha
        while iter < self.max_iter and error > self.tolerance and (alpha > 0).all():
            alpha_old = alpha
            g = -n_doc * (
                sp.special.digamma(alpha_old) - sp.special.digamma(alpha_old.sum())
            ) + (
                sp.special.digamma(gamma)
                - sp.special.digamma(gamma.sum(axis=1, keepdims=True))
            ).sum(
                axis=0
            )
            h = -n_doc * sp.special.polygamma(1, alpha)
            z = sp.special.polygamma(1, alpha.sum())
            c = (g / h).sum() / (np.reciprocal(z) + np.reciprocal(h).sum())
            alpha = alpha_old - (g - c) / h

            iter += 1
            error = (np.abs(alpha_old - alpha) / np.abs(alpha_old)).sum()

        if np.isnan(alpha).any() or (alpha <= 0).any():
            self._update_alpha(init_alpha * 2 + 1e-3, gamma, n_doc)

        return alpha, iter
