"""Common variable and function for data submodule."""

import abc


class TopicModel(abc.ABC):
    """Abstract class for topic modelling algorithms."""

    @abc.abstractmethod
    def fit(self, data):
        """Find all the relevant parameters to get
        the distribution of the topic among the documents.

        Args:
            data: input data, genrally a matrix of terms and documents.
        """
        pass

    @abc.abstractmethod
    def transform(self, data):
        """Retrieve the distribution of the topic among the documents from prevously
        computed parameters.

        Args:
            data: input data, genrally a matrix of terms and documents.
        """
        pass

    @abc.abstractmethod
    def fit_transform(self, data):
        """Retrieve the distribution of the topic among the documents, computing the
        parameters at the same time.

        This is to be preferred to calling :func:`~topicmodel.common.TopicModel.fit`
        and :func:`~topicmodel.common.TopicModel.transform` subsequently.

        Args:
            data: input data, genrally a matrix of terms and documents.
        """
        pass
