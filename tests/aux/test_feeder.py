import unittest
import torch
import numbers

import aux.feeder as feeder
from tests.aux.fixtures.simple_rnn import SimpleRNN


class TestFeeder(unittest.TestCase):

    def test_feed_with_normalizing(self):
        forward_function = SimpleRNN(input_size=2).forward
        batch = [
            (torch.FloatTensor([[1, 0], [1, 1]]), torch.FloatTensor([1, 1, 0, 0])),
            (torch.FloatTensor([[1.0, 1], [0.75, 1], [0.5, 1], [0, 1]]), torch.FloatTensor([0.5])),
            (torch.FloatTensor([[2, 2], [1, 2], [0, 2], [-2, 2]]), torch.FloatTensor([0]))
        ]

        actual_y, actual_y_denormalized = feeder.feed(
            batch,
            forward_function,
            variable_length=True,
            x_requires_grad=True,
            normalize=True
        )

        self.assertIsInstance(actual_y, torch.FloatTensor)
        self.assertIsInstance(actual_y_denormalized, torch.FloatTensor)
        self.assertEqual(actual_y.shape, (3, 4, 1))
        self.assertEqual(actual_y_denormalized.shape, (3, 4, 1))
        self.assertEqual(actual_y[0][0][0], actual_y[1][0][0])
        self.assertNotEqual(actual_y_denormalized[0][0][0], actual_y_denormalized[1][0][0])
        self.assertEqual(actual_y[0][3][0], actual_y[1][3][0])
        self.assertNotEqual(actual_y_denormalized[0][3][0], actual_y_denormalized[1][3][0])

    def test_feed_loss_with_normalizing(self):
        forward_function = SimpleRNN(input_size=2).forward
        loss_function = torch.nn.MSELoss(reduce=False)
        batch = [
            (torch.FloatTensor([[1, 0], [1, 1]]), torch.FloatTensor([[1], [1], [0], [0]])),
            (torch.FloatTensor([[1.0, 1], [0.75, 0], [0.5, 1], [0, 1]]), torch.FloatTensor([[0.5]])),
            (torch.FloatTensor([[2, 1], [1, 1], [0, 1], [-2, 0]]), torch.FloatTensor([[0]]))
        ]

        loss = feeder.feed(
            batch,
            forward_function,
            loss_function=loss_function,
            variable_length=True,
            x_requires_grad=True,
            normalize=True
        )[0]

        self.assertIsInstance(loss, torch.autograd.Variable)
        self.assertEqual(loss.shape, (10, 1))


    # def test_normalize_and_denormalize_with_3_dimensional_input(self):
    #     x = torch.autograd.Variable(
    #         torch.FloatTensor([
    #             [[1.0], [0.75], [0.5], [0]],
    #             [[2], [1], [0], [-2]],
    #         ])
    #     )
    #     expected_normalized_x = torch.autograd.Variable(
    #         torch.FloatTensor([
    #             [[1.0], [0.75], [0.5], [0]],
    #             [[1.0], [0.75], [0.5], [0]]
    #         ])
    #     )
    #
    #
    #     normalized_x, mins, maxs = feeder.__normalize__(x)
    #     self.assertTrue(torch.equal(normalized_x, expected_normalized_x))
    #
    #     y = torch.autograd.Variable(
    #         torch.FloatTensor([
    #             [0.75, 0.75, 0.75],
    #             [0.75, 0.75, 0.75]
    #         ])
    #     )
    #     expected_denormalized_y = torch.autograd.Variable(
    #         torch.FloatTensor([
    #             [0.75, 0.75, 0.75],
    #             [1, 1, 1]
    #         ])
    #     )
    #     denormalized_y = feeder.__denormalize__(y, mins, maxs)
    #     self.assertTrue(torch.equal(denormalized_y, expected_denormalized_y))

if __name__ == '__main__':
    unittest.main()
