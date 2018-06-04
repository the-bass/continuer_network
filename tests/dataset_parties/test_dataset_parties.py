import unittest
import dataset_parties
from tests.dataset_parties.fixtures.trivial_dataset import TrivialDataset


class TestDatasetParties(unittest.TestCase):

    def test_ng_style(self):
        dataset = TrivialDataset(set_size=100)
        train_set, dev_set, test_set = dataset_parties.ng_style(dataset)
        self.assertEqual(len(train_set), 60)
        self.assertEqual(len(dev_set), 20)
        self.assertEqual(len(test_set), 20)

        dataset = TrivialDataset(set_size=50010)
        train_set, dev_set, test_set = dataset_parties.ng_style(dataset)
        self.assertEqual(len(train_set), 30010)
        self.assertEqual(len(dev_set), 10000)
        self.assertEqual(len(test_set), 10000)

if __name__ == '__main__':
    unittest.main()
