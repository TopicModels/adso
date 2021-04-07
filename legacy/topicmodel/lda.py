"""Latent Dirichlet Allocation.

From Blei et al. 2003.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import scipy as sp
import sparse

from .common import TopicModel


class LDA(TopicModel):
    """Latent Dirichlet Allocation.

    Implemented following Blei, Jordan, Ng 2003.
    """

    def __init__(
        self: LDA,
        n_topic: int = 10,
        max_iter: int = 200,
        tolerance: float = 0,
        epsilon: float = 1e-50,
    ) -> None:
        """Initialize the algorithm.

        Args:
            n_topic (int, optional): number of topics in copus to be estimated.
                Defaults to 10.
            max_iter (int, optional): max number of iterations for iterative steps.
                Defaults to 200.
            tolerance (float, optional): target relative error for early stopping
                in iterative steps. Defaults to 0.
            epsilon (float, optional): small number to avoid -inf in logarithm result.
                Defaults to 1e-50.
        """
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
        """Estimate LDA parameters, particularly alpha e beta.

        Args:
            data (sp.sparse.spmatrix): document-term matrix with counts as entry.
                For example the output of :class:`CountVectorizer`

        Returns:
            Tuple[np.array, np.array, np.array, np.array, float, int]:
                alpha (np.array):
                    Dirichlet priors for LDA model (also stored as instance attributes)
                beta (np.array):
                    Multinomial priors for LDA model (also stored as instance
                    attributes)
                gamma (np.array):
                    Dirichlet priors for auxiliary model
                phi (np.array):
                    Multinomial priors for auxiliary model
                likelihood (float):
                    achieved likelihood
                iter (int):
                    number of iteration of EM algorithm
        """
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

        phi = sparse.full((n_doc, n_term, n_topic), 1 / n_topic)
        gamma = (
            alpha[np.newaxis, :] + data.sum(axis=1, keepdims=True).todense() / n_topic
        )

        loglikelihood = self._loglikelihood(data, alpha, beta, gamma, phi)
        iter: int = 0
        error: float = 1

        while iter < self.max_iter and error > self.tolerance:

            loglikelihood_old = loglikelihood

            gamma, phi = self._estep(alpha, beta, gamma)
            alpha, beta, _ = self._mstep(data, alpha, gamma, phi)

            loglikelihood = self._loglikelihood(data, alpha, beta, gamma, phi)

            error = abs((loglikelihood_old - loglikelihood) / loglikelihood_old)
            iter += 1
            print(f"Iteration {iter} Log-Likelihood {loglikelihood}")

        beta = beta.tocsr()

        self.alpha = alpha
        self.beta = beta

        return alpha, beta, gamma, phi, loglikelihood, iter

    def transform(self: LDA, data: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
        """Estimate the probability of each topic in each document, given the model.

        Must be called after a :func:`~topicmodel.LDA.fit` method on the same instance.

        Args:
            data (sp.sparse.spmatrix): document-term matrix with counts as entry.
                For example the output of :class:`CountVectorizer`

        Returns:
            sp.sparse.spmatrix: the probability of each document (rows) given the
                different topics (columns).
        """
        return sp.sparse.csr_matrix(
            np.exp(data @ np.log(self.beta.T.todense() + self.epsilon))
        )

    def fit_transform(
        self: LDA, data: sp.sparse.spmatrix
    ) -> Tuple[sp.sparse.matrix, np.array, np.array, np.array, np.array, float, int]:
        """Estimate the probability of each topic in each document.

        Perform the estimation of both model's parameters (:func:`~topicmodel.LDA.fit`)
        and topic-document relation (:func:`~topicmodel.LDA.transform`).

        Args:
            data (sp.sparse.spmatrix): document-term matrix with counts as entry.
                For example the output of :class:`CountVectorizer`

        Returns:
            Tuple[sp.sparse.matrix, np.array, np.array, np.array, np.array, float, int]:
                estimation (sp.sparse.spmatrix):
                    the probability of each document (rows) given the different topics
                    (columns).
                alpha (np.array):
                    Dirichlet priors for LDA model (also stored as instance attributes)
                beta (np.array):
                    Multinomial priors for LDA model (also stored as instance
                    attributes)
                gamma (np.array):
                    Dirichlet priors for auxiliary model
                phi (np.array):
                    Multinomial priors for auxiliary model
                likelihood (float):
                    achieved likelihood
                iter (int):
                    number of iteration of EM algorithm
        """
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
        """Estimate a lower bound for the log-likelihood, using Jensen's inequality
        and auxiliary model.

        Args:
            data (sp.sparse.matrix): document-term matrix with counts as entry.
                For example the output of :class:`CountVectorizer`

        Returns:
            float: the computed lower bound for the loglikelihood.
        """
        n_doc, n_term = data.shape
        sum_digamma = sp.special.digamma(gamma) - sp.special.digamma(
            gamma.sum(axis=1, keepdims=True)
        )  # di
        likelihood = sp.special.loggamma(alpha.sum()) * n_doc
        likelihood -= sp.special.loggamma(alpha).sum() * n_doc
        likelihood += ((alpha - 1) * (sum_digamma.sum(axis=0))).sum()
        likelihood += (phi.sum(axis=1) * sum_digamma).sum()
        likelihood += (
            phi
            * sparse.elemwise(np.nan_to_num, np.log(beta + self.epsilon))[
                :, :, np.newaxis
            ].T
            * data[:, :, np.newaxis]
        ).sum()
        likelihood -= sp.special.loggamma(gamma.sum(axis=1)).sum()
        likelihood += sp.special.loggamma(gamma).sum()
        likelihood -= ((gamma - 1) * sum_digamma).sum()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r".*((log)|(multiply)).*", category=RuntimeWarning
            )
            likelihood -= sparse.elemwise(np.nan_to_num, phi * np.log(phi)).sum()
        return likelihood

    def _estep(
        self: LDA,
        alpha: np.array,
        beta: np.array,
        gamma: np.array,
    ) -> Tuple[np.array, np.array]:
        """Estimating the auxiliary model parameters to minimize the KL-divergence
        between the auxiliary model and the last estimation of LDA model given
        the data.

        Args:
            alpha (np.array): estimation of Dirichlet parameters for the LDA model
            beta (np.array): estimation of multinomial parameters for the LDA model
            gamma (np.array): last estimation of Dirichlet parameters for the
                auxiliary model

        Returns:
            Tuple[np.array, np.array]: gamma (Dirichlet parameters) and phi
                (multinomial parameters) for auxiliary model.
                Note that gamma and phi are document specific.
        """
        phi = (
            np.exp(sp.special.digamma(gamma))[:, np.newaxis, :]
            * beta[:, :, np.newaxis].T
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r".*(true_divide).*", category=RuntimeWarning
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
    ) -> Tuple[np.array, np.array, float]:
        """Estimating the LDA parameters with maximum likelihood estimation based
        on data and approximate posterior distribution given by the auxiliary model.

        Args:
            data (sp.sparse.matrix): document-term matrix with counts as entry.
                For example the output of :class:`CountVectorizer`
            alpha (np.array): last estimation of Dirichlet parameters for the LDA model
            gamma (np.array): estimation of Dirichlet parameters for the
                auxiliary model
            phi (np.array): estimation of multinomial parameters for the
                auxiliary model

        Returns:
            Tuple[np.array, np.array, int]: alpha (Dirichlet parameters) and beta
                (multinomial parameters) for LDA model and the number of iterations
                computed for the estimation of alpha with a Newton-like method.
        """
        beta = (phi * data[:, :, np.newaxis]).sum(axis=0).T
        beta = beta / beta.sum(axis=1, keepdims=True).todense()
        alpha, iter_alpha = self._update_alpha(alpha, gamma)

        return alpha, beta, iter_alpha

    def _update_alpha(
        self: LDA, alpha: np.array, gamma: np.array
    ) -> Tuple[np.array, int]:
        """Modified Newton-Raphson method to update Diriclet priors for LDA.

        Args:
            alpha (np.array): inital alpha
            gamma (np.array): parameters for Dirichlet priors of auxiliary model

        Returns:
            Tuple[np.array, int]: return updated alpha and the number of iteration.
        """
        iter = 0
        error = 1
        init_alpha = alpha
        n_doc = gamma.shape[0]
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
            self._update_alpha(init_alpha * 2 + 1e-3, gamma)

        return alpha, iter
