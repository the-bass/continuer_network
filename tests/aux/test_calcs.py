import unittest
import torch

from aux.calcs import triangle_costs, l1_costs
from fixtures.simple_fc import SimpleFC
from fixtures.mini_dataset import MiniDataset


class TestCalcs(unittest.TestCase):

    def test_triangle_costs(self):
        network = SimpleFC()
        dataset = MiniDataset()

        losses = triangle_costs(network, dataset, batch_size=1)
        self.assertEqual(losses.shape, (2, 1))

    def test_l1_costs(self):
        network = SimpleFC()
        dataset = MiniDataset()

        losses = l1_costs(network, dataset, batch_size=1)
        self.assertEqual(losses.shape, (2, 1))

if __name__ == '__main__':
    unittest.main()
