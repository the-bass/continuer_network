import unittest
import torch
import numbers

from networks.simple_fc import SimpleFC


class TestSimpleFC(unittest.TestCase):

    def test_forward(self):
        training_set = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        ]
        input = torch.autograd.Variable(
            torch.FloatTensor(training_set)
        )

        network = SimpleFC(input_size=20)
        output = network.forward(input)

        self.assertIsInstance(output, torch.autograd.Variable)
        self.assertEqual(output.shape, (2, 1))
        self.assertIsInstance(float(output[0]), numbers.Number)
        self.assertIsInstance(float(output[1]), numbers.Number)

    def test_forward_sample(self):
        sample = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        network = SimpleFC(input_size=20)
        output = network.forward_sample(sample)

        self.assertIsInstance(output, float)

if __name__ == '__main__':
    unittest.main()
