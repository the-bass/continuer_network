import torch.utils.data
import sequences_creator


class NumbersDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.records = self.__load_dataset__()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @staticmethod
    def __load_dataset__():
        with open('./data/numbers.txt') as file:
            count = 0
            for line in file:
                numbers = line.split(',')
                count += 1
