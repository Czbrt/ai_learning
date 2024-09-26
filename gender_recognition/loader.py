import numpy as np
from torch.utils.data import Dataset


def load_data_from_txt(txt_file):
    data = []
    labels = []

    with open(txt_file, 'r') as f:
        for line in f:
            height, weight, label = map(float, line.split())
            data.append([height, weight])  # 身高和体重作为特征
            labels.append(label)  # 标签

    return np.array(data), np.array(labels)


class GenderDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data, self.labels = load_data_from_txt(txt_file)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.data)
