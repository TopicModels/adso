"""Dataset class.

Define data-container for other classes.
"""

from __future__ import annotations

from typing import List, Tuple


class Dataset:
    """Dataset class."""

    def __init__(self: Dataset, data: List[str]) -> None:
        """Constructor for Dataset class.

        Args:
            data (List[str]): A list of string, each will be considered as one document.
        """
        self.data: List[str] = data

    def get_data(self: Dataset) -> List[str]:
        """Access data stored into the dataset.

        Returns:
            List[str]: Return a list of string, each one is a document
        """
        return self.data

    def __len__(self: Dataset) -> int:
        """Return the number of the document in the dataset.

        Returns:
            int: number of document in the dataset
        """
        return len(self.data)

    def __getitem__(self: Dataset, index: int) -> str:
        """Enable the data[i] syntax to retrieve the data.

        Args:
            index (int): index of the required document

        Returns:
            str: the required document as string
        """
        return self.data[index]

    def __add__(self: Dataset, other: Dataset) -> Dataset:
        """Overload + operator to concatenate dataset.

        Args:
            other (Dataset): the second dataset to be concatenated

        Returns:
            Dataset: a dataset with the documents of both summed datasets
        """
        return Dataset(self.get_data() + other.get_data())


class LabelledDataset(Dataset):
    """Labbelled Dataset class.

    A subclass of :class:`Dataset` wich store also labels for documents.
    Documents and labels are stored in two different lists where each pair share
    the index.
    """

    def __init__(self: LabelledDataset, data: List[Tuple[str, str]]) -> None:
        """Contructor for LabelledDataset class.

        Args:
            data (List[Tuple[str, str]]): list of (document, label) tuples of strings.
        """
        self.data, self.labels = zip(*data)

    def get_data(self: LabelledDataset) -> List[str]:
        """Access data stored into the dataset.

        Returns:
            List[str]: Return a list of string, each one is a document
        """
        return self.data

    def get_labels(self: LabelledDataset) -> List[str]:
        """Access labels stored into the dataset.

        Returns:
            List[str]: Return a list of string, each one is a label
        """
        return self.labels

    def toDataset(self: LabelledDataset) -> Dataset:
        """Convert a labelledDataset to Dataset.

        Returns:
            Dataset: a Dataset with the same documents list but without labels.
        """
        return Dataset(self.get_data())

    def __getitem__(self: LabelledDataset, index: int) -> Tuple[str, str]:
        """Enable the data[i] syntax to retrieve the data.

        Args:
            index (int): index of the required document

        Returns:
            Tuple[str, str]: the required document and its label, as tuple
        """
        return self.data[index], self.labels[index]

    def __add__(self: LabelledDataset, other: Dataset) -> Dataset:
        """Overload + operator to concatenate dataset.

        Args:
            other (Dataset): the second dataset to be concatenated
                (must be a Dataset subclass)

        Returns:
            Dataset: a dataset with the documents of both summed dataset.
                If both Dataset are Labelled, instead, return a LabelledDataset
                with also the labels.
        """
        if isinstance(other, LabelledDataset):
            return LabelledDataset(
                list(zip(self.get_data(), self.get_labels()))
                + list(zip(other.get_data(), other.get_labels()))
            )
        else:
            return Dataset(self.get_data() + other.get_data())
