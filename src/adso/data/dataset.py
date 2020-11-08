from __future__ import annotations

import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self: Dataset, data: list[str]) -> None:
        self.data: list(str) = data

    def get_data(self: Dataset) -> list[str]:
        return self.data

    def __len__(self: Dataset):
        return len(self.data)

    def __getitem__(self: Dataset, index: int):
        return self.data[index]


class LabelledDataset(Dataset):
    def __init__(self: LabelledDataset, data: list[tuple[str, str]]) -> None:
        self.data, self.label = zip(*data)

    def get_data(self: LabelledDataset) -> list[str]:
        return self.data

    def get_label(self: LabelledDataset) -> list[str]:
        return self.label

    def __getitem__(self: Dataset, index: int):
        return self.data[index], self.label[index]
