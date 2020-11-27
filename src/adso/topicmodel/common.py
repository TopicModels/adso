import abc


class TopicModel(abc.ABC):
    @abc.abstractmethod
    def fit(data):
        pass

    @abc.abstractmethod
    def transform(data):
        pass

    @abc.abstractmethod
    def fit_transform(data):
        pass
