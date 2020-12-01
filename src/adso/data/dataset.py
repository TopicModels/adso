from __future__ import annotations

from typing import List, Tuple


class Dataset:
    def __init__(self: Dataset, data: List[str]) -> None:
        # add a check type on data elements?
        self.data: List(str) = data

    def get_data(self: Dataset) -> List[str]:
        return self.data

    def __len__(self: Dataset):
        return len(self.data)

    def __getitem__(self: Dataset, index: int):
        return self.data[index]

    def __add__(self: Dataset, other: Dataset):
        return Dataset(self.get_data() + other.get_data())


class LabelledDataset(Dataset):
    def __init__(self: LabelledDataset, data: List[Tuple[str, str]]) -> None:
        self.data, self.labels = zip(*data)

    def get_data(self: LabelledDataset) -> List[str]:
        return self.data

    def get_labels(self: LabelledDataset) -> List[str]:
        return self.labels

    def __getitem__(self: Dataset, index: int):
        return self.data[index], self.labels[index]

    def __add__(self: LabelledDataset, other: LabelledDataset):
        return LabelledDataset(
            list(zip(self.get_data(), self.get_labels()))
            + list(zip(other.get_data(), other.get_labels()))
        )
