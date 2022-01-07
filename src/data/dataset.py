import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class TwitterDisaster(Dataset):
    def __init__(self, dir, train):
        path = dir + '/train.csv' if train else data + '/test.csv'
        files = pd.read_csv(path)
        self.x = files[files.columns[:-1]]
        self.y = files[files.columns[-1]]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]
