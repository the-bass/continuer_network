import unittest
import torch
from networks.continuer import Continuer


class TestContinuer(unittest.TestCase):

    def test_forward(self):
        sequences = [
            torch.autograd.Variable(
                torch.FloatTensor([
                    [56], [-1], [0]
                ])
            ),
            torch.autograd.Variable(
                torch.FloatTensor([
                    [0], [0]
                ])
            )
        ]
        x = torch.nn.utils.rnn.pack_sequence(sequences)

        # The shape of the input sequences should be (n, 2).
        self.assertEqual(sequences[0].shape, (3, 1))

        network = Continuer()
        y = network.forward(x)

        self.assertIsInstance(y, torch.FloatTensor)
        self.assertEqual(y.shape, (2, 1))

if __name__ == '__main__':
    unittest.main()
