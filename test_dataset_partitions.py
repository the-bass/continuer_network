import unittest
from dataset_partitions import train_set, dev_set, test_set
from aux import hashed_dataset # WARNING: Use hashed_dataset on a huge set takes f***** long!
import itertools


class TestDatasetPartitions(unittest.TestCase):

    def test_train_set(self):
        self.assertEqual(len(train_set), 30000)

    def test_dev_set(self):
        self.assertEqual(len(dev_set), 10000)

    def test_test_set(self):
        self.assertEqual(len(test_set), 10000)

    def test_sets_disjunct(self):
        """ Test that the datasets have no samples in common. """

        # To speed things up, compare lists of hashes.
        hashed_train_set = hashed_dataset(train_set)
        hashed_dev_set = hashed_dataset(dev_set)
        hashed_test_set = hashed_dataset(test_set)

        train_set_set = set(hashed_train_set)
        dev_set_set = set(hashed_dev_set)
        test_set_set = set(hashed_test_set)

        self.assertEqual(len(train_set_set.intersection(dev_set_set)), 0)
        self.assertEqual(len(dev_set_set.intersection(test_set_set)), 0)
        self.assertEqual(len(train_set_set.intersection(test_set_set)), 0)

if __name__ == '__main__':
    unittest.main()
