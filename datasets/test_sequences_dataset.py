import unittest
import torch
from datasets.sequences_dataset import SequencesDataset


class TestSequencesDataset(unittest.TestCase):

    def test___len__(self):
        dataset = SequencesDataset()
        self.assertEqual(len(dataset), 50000)

    def test___getitem__(self):
        dataset = SequencesDataset()
        samples = dataset[:2]
        self.assertIsInstance(samples[0], tuple)
        self.assertIsInstance(samples[1], tuple)

    def test_format_of_x(self):
        dataset = SequencesDataset()
        x, y = dataset[0]

        self.assertIsInstance(x, torch.FloatTensor)
        self.assertEqual(len(x.shape), 2)
        self.assertGreater(x.shape[0], 8)
        self.assertEqual(x.shape[1], 1)

    def test_format_of_y(self):
        dataset = SequencesDataset()
        x, y = dataset[0]

        self.assertIsInstance(y, torch.FloatTensor)
        self.assertEqual(y.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
