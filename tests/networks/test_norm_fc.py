import unittest
import torch
import numbers

from networks.norm_fc import NormFC


class TestNormFC(unittest.TestCase):

    def test_forward(self):
        training_set = [
            [1.5, 1.25, 0.5, 1],
            [1500, 1250, 500, 1000],
        ]
        input = torch.autograd.Variable(
            torch.FloatTensor(training_set)
        )

        network = NormFC(input_size=4)
        output = network.forward(input)

        self.assertIsInstance(output, torch.autograd.Variable)
        self.assertEqual(output.shape, (2, 1))
        self.assertIsInstance(output.data[0][0], numbers.Number)
        self.assertIsInstance(output.data[1][0], numbers.Number)

    def test_normalization(self):
        training_set = [
            [1.5, 1.25, 0.5, 1],
            [1500, 1250, 500, 1000],
        ]
        input = torch.autograd.Variable(
            torch.FloatTensor(training_set)
        )

        network = NormFC(input_size=4)
        output = network.forward(input)

        self.assertAlmostEqual(float(output[0]), float(output[1]) / 1000, places=6)

if __name__ == '__main__':
    unittest.main()
