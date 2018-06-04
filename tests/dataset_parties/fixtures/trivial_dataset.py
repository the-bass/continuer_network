import torch.utils.data


class TrivialDataset(torch.utils.data.Dataset):

    def __init__(self, set_size=1000):
        self.records = list(map(
            lambda x: (list(range(x + 1, x + 6)), list(range(x + 2, x + 4))),
            list(range(set_size))
        ))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        x = torch.FloatTensor(record[0])
        y = torch.FloatTensor(record[1])

        return {'x': x, 'y': y}
