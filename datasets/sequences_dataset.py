import torch.utils.data
import sequences_creator


class SequencesDataset(torch.utils.data.Dataset):

    def __init__(self, set_size=50000):
        self.records = self.__create_dataset__(set_size=set_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @staticmethod
    def __create_dataset__(set_size):
        if set_size:
            dataset = sequences_creator.create_dataset(set_size=set_size)
        else:
            dataset = sequences_creator.create_dataset()

        # Convert numpy arrays to torch tensors.
        dataset = list(map(
            lambda sample: (torch.from_numpy(sample[0]).float(), torch.from_numpy(sample[1]).float()),
            dataset
        ))

        dataset = list(map(
            lambda element: (element[0].view(-1, 1), element[1].view(-1, 1)),
            dataset
        ))

        return dataset
