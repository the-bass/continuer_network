import unittest
import torch
import numpy

from datasets.one_hole_training import OneHoleTrainingDataset


class TestOneHoleTrainingDataset(unittest.TestCase):

    def setUp(self):
        self.embedding_layers = 3
        self.dataset = OneHoleTrainingDataset(embedding_layers=self.embedding_layers)

    # def test___find_holes__(self):
    #     embedding_layers = 3
    #     list = numpy.array([
    #         1, 2, None, 4, 5, 6, None, 8, 9, 10, None, 11, 12, 13,
    #         None, 14, 15, None, None, 16, 17, 18, None, 19, 20, 21
    #     ], dtype='float64')
    #
    #     self.assertEqual(
    #         self.dataset.__find_holes__(embedding_layers, list),
    #         [
    #             [4, 5, 6, 8, 9, 10],
    #             [8, 9, 10, 11, 12, 13],
    #             [16, 17, 18, 19, 20, 21]
    #         ]
    #     )

    # def test___training_pairs_from_list__(self):
    #     embedding_layers = 3
    #     dataset = OneHoleTrainingDataset(embedding_layers=embedding_layers, currency='Euro')
    #     disjunct = False
    #
    #     liszt = numpy.array([
    #         None, 1, 2, 3, 4, 5, 6, 7, 8, None, 5, 6, 7, 8, 9, 10, None,
    #         11, 12, 13, 14, 15, 16, 17, None
    #     ], dtype='float64')
    #
    #     training_pairs = dataset.__training_pairs_from_list__(embedding_layers, liszt, disjunct)
    #
    #     self.assertEqual(len(training_pairs), 3)
    #     self.assertEqual(
    #         training_pairs,
    #         [
    #             ([1, 2, 3, 5, 6, 7], 4),
    #             ([2, 3, 4, 6, 7, 8], 5),
    #             ([11, 12, 13, 15, 16, 17], 14),
    #
    #         ]
    #     )
    #
    # def test_len(self):
    #     dataset = OneHoleTrainingDataset(embedding_layers=100, currency='Euro')
    #     self.assertGreater(len(dataset), 300)
    #
    # def test_indexing(self):
    #     dataset = OneHoleTrainingDataset(embedding_layers=7, currency='Euro')
    #
    #     # Check first example.
    #     sample = dataset[0]
    #     torch.eq(sample['x'], torch.FloatTensor([
    #         1.18398,
    #         1.18373,
    #         1.17525,
    #         1.1711,
    #         1.17706,
    #         1.17706,
    #         1.1667100000000001,
    #         1.16248,
    #         1.1606,
    #         1.1606,
    #         1.1736799999999998,
    #         1.17468,
    #         1.1744,
    #         1.1693799999999999
    #     ]))
    #     self.assertEqual(sample['y'], 1.1624)
    #
    #     # Check last example.
    #     sample = dataset[-1]
    #     torch.eq(sample['x'], torch.FloatTensor([
    #         1.2291,
    #         1.2302,
    #         1.2378,
    #         1.2369,
    #         1.2341,
    #         1.2301,
    #         1.2309,
    #         1.2286,
    #         1.2316,
    #         1.2346,
    #         1.2411,
    #         1.2376,
    #         1.2398,
    #         1.2321
    #     ]))
    #     self.assertEqual(sample['y'], 1.2276)
    #
    # def test_samples_disjunct(self):
    #     """
    #     Check behaviour for `disjunct = False`.
    #     """
    #     dataset = OneHoleTrainingDataset(embedding_layers=2, currency='Euro')
    #
    #     # Check first example.
    #     sample = dataset[0]
    #     torch.eq(sample['x'], torch.FloatTensor([
    #         1.18398,
    #         1.18373,
    #         1.1711,
    #         1.17706
    #     ]))
    #     self.assertEqual(sample['y'], 1.17525)
    #
    #     # Check second example.
    #     sample = dataset[1]
    #     torch.eq(sample['x'], torch.FloatTensor([
    #         1.18373,
    #         1.17525,
    #         1.17706,
    #         1.17706
    #     ]))
    #     self.assertEqual(sample['y'], 1.1711)
    #
    #     """
    #     Check behaviour for `disjunct = True`.
    #     """
    #     dataset = OneHoleTrainingDataset(embedding_layers=2, currency='Euro', disjunct=True)
    #
    #     # Check first example.
    #     sample = dataset[0]
    #     torch.eq(sample['x'], torch.FloatTensor([
    #         1.18398,
    #         1.18373,
    #         1.1711,
    #         1.17706
    #     ]))
    #     self.assertEqual(sample['y'], 1.17525)
    #
    #     # Check second example.
    #     sample = dataset[1]
    #     torch.eq(sample['x'], torch.FloatTensor([
    #         1.17706,
    #         1.1667100000000001,
    #         1.16248,
    #         1.16060
    #     ]))
    #     self.assertEqual(sample['y'], 1.16240)
    #
    # def test_data_does_not_contain_none(self):
    #     dataset = OneHoleTrainingDataset(embedding_layers=3, currency=None)
    #
    #     for x, y in dataset:
    #         for value in x:
    #             self.assertIsNotNone(value)
    #         self.assertIsNotNone(y)

    def test_normalize_list(self):
        list = [1005, 1090, 902, 1000, 947]

        normalized_list = OneHoleTrainingDataset.normalize_list(list)

        self.assertEqual(normalized_list, [1.1, 1.2, 0.9, 1, 0.95])

if __name__ == '__main__':
    unittest.main()
