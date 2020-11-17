from __future__ import annotations

from typing import List, Tuple


class Dataset:
    def __init__(self: Dataset, data: List[str]) -> None:
        # add a chack type on data elements?
        self.data: List(str) = data

    def get_data(self: Dataset) -> List[str]:
        return self.data

    def __len__(self: Dataset):
        return len(self.data)

    def __getitem__(self: Dataset, index: int):
        return self.data[index]


class LabelledDataset(Dataset):
    def __init__(self: LabelledDataset, data: List[Tuple[str, str]]) -> None:
        self.data, self.label = zip(*data)

    def get_data(self: LabelledDataset) -> List[str]:
        return self.data

    def get_label(self: LabelledDataset) -> List[str]:
        return self.label

    def __getitem__(self: Dataset, index: int):
        return self.data[index], self.label[index]
