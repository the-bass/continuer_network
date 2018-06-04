import torch.utils.data


class MiniDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.data = [
            {'x': torch.FloatTensor([1, 2, 4, 5]), 'y': 3},
            {'x': torch.FloatTensor([5, 6, 8, 9]), 'y': 7}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
