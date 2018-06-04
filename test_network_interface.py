import unittest
import torch_testing as tt
import torch
from network_interface import NetworkInterface
from datasets.sequences_dataset import SequencesDataset
from networks.continuer import Continuer


class TestNetworkInterface(unittest.TestCase):

    def setUp(self):
        self.network = Continuer()
        self.interface = NetworkInterface(network=self.network)

    def test_performance_on_set(self):
        dataset = SequencesDataset(set_size=10)

        performance = self.interface.performance_on_set(dataset)

        self.assertIsInstance(performance, float)

    def test_train_one_epoch(self):
        dataset = SequencesDataset(set_size=10)

        mean_loss = self.interface.train_one_epoch(
            dataset,
            weight_decay=0,
            learning_rate=0.02,
            shuffle=False
        )

        self.assertIsInstance(mean_loss, float)

    def test_performance_on_samples(self):
        dataset = SequencesDataset(set_size=10)

        performance = self.interface.performance_on_samples(dataset)
        sample = performance[0]
        # raw_sample = dataset[0]

        self.assertIsInstance(sample.x, torch.FloatTensor)
        self.assertEqual(len(sample.x.shape), 2)
        self.assertGreater(sample.x.shape[0], 8)
        self.assertEqual(sample.x.shape[1], 1)

        self.assertIsInstance(sample.loss, float)
        self.assertIsInstance(sample.actual_y, float)
        self.assertIsInstance(sample.predicted_y, float)

    def test___preprocessed_batch__(self):
        batch_size = 5
        batch = SequencesDataset(set_size=batch_size)[:batch_size]

        x, y, normalization_factors, normalization_offsets = self.interface.__preprocessed_batch__(batch)

        self.assertIsInstance(x, torch.nn.utils.rnn.PackedSequence)
        x_unpacked, x_lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        self.assertEqual(x_unpacked.shape[0], batch_size)
        self.assertGreater(x_unpacked.shape[1], 8)
        self.assertEqual(x_unpacked.shape[2], 1)

        self.assertIsInstance(y, torch.FloatTensor)
        # y_unpacked, y_lengths = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        self.assertEqual(y.shape, (batch_size, 1))
        self.assertFalse(y.requires_grad)

        self.assertIsInstance(normalization_factors, torch.FloatTensor)
        self.assertEqual(normalization_factors.shape, (batch_size,))

        self.assertIsInstance(normalization_offsets, torch.FloatTensor)
        self.assertEqual(normalization_offsets.shape, (batch_size,))

    # def test_plugin_known_values(self):
    #     interface = NetworkInterface()
    #     y = [
    #         torch.tensor([[5], [8], [1]]),
    #         torch.tensor([[9], [1]])
    #     ]
    #     x = [
    #         torch.tensor([[4, 1], [9, 0], [2, 0]]),
    #         torch.tensor([[3, 1], [2, 1]])
    #     ]
    #
    #     y = torch.nn.utils.rnn.pack_sequence(y)
    #     x = torch.nn.utils.rnn.pack_sequence(x)
    #
    #     y = interface.__plugin_known_values__(y, x)
    #
    #     y_unpacked, y_lengths = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
    #     tt.assert_equal(y_lengths, torch.tensor([3, 2]))
    #     tt.assert_equal(y_unpacked, torch.tensor([
    #         [[4], [8], [1]],
    #         [[3], [2], [0]]
    #     ]))

    def test_normalize(self):
        batch = [
            (
                torch.FloatTensor([[1.0], [0.75], [0.5], [0.5]]),
                torch.FloatTensor([[2]])
            ),
            (
                torch.FloatTensor([[1.5], [2], [1.5], [0], [-2]]),
                torch.FloatTensor([[6]])
            )
        ]

        expected_offsets = torch.FloatTensor([-0.5, 2])
        expected_factors = torch.FloatTensor([2, 0.25])
        expected_normalized_batch = [
            (
                torch.FloatTensor([[1], [0.5], [0], [0]]),
                torch.FloatTensor([[3]])
            ),
            (
                torch.FloatTensor([[3.5/4], [1], [3.5/4], [0.5], [0]]),
                torch.FloatTensor([[2]])
            )
        ]

        normalized_batch, factors, offsets = self.interface.__normalize_batch__(batch)
        tt.assert_equal(offsets, expected_offsets)
        tt.assert_equal(factors, expected_factors)
        for i, normalized_element in enumerate(normalized_batch):
            normalized_x, normalized_y = normalized_element
            expected_normalized_x, expected_normalized_y = expected_normalized_batch[i]

            tt.assert_equal(normalized_x, expected_normalized_x)
            tt.assert_equal(normalized_y, expected_normalized_y)

    # def test_denormalize(self):
    #     """NOTE: This test only works under the condition that the `__normalize__` function works as expected."""
    #
    #     interface = NetworkInterface()
    #
    #     batch = [
    #         (
    #             torch.FloatTensor([[1.0], [0.75], [0.5], [0.5]]),
    #             torch.FloatTensor([[52, 23, 1], [0.5, 30.0, 1], [-13.5, 56.2, -13]])
    #         ),
    #         (
    #             torch.FloatTensor([[1.5, 7], [1, 3], [0, 6], [0, 2.4], [-2, -0.25]]),
    #             torch.FloatTensor([[2, 3, 4], [0.5, 0.0, 1], [-1.5, 5.2, 3]])
    #         )
    #     ]
    #     normalized_tensor, factors, offsets = interface.__normalize_batch__(batch)
    #
    #     normalized_tensor = torch.cat([
    #         normalized_tensor[0][1].view(1, *normalized_tensor[0][1].shape),
    #         normalized_tensor[1][1].view(1, *normalized_tensor[1][1].shape)
    #     ])
    #
    #     denormalized_tensor = interface.__denormalize_tensor__(normalized_tensor, factors, offsets)
    #
    #     round_to = 5
    #     round_factor = 10**round_to
    #     rounded_expected_tensor = torch.round(batch[1][1] * round_factor) / round_factor
    #     rounded_actual_tensor = torch.round(denormalized_tensor[1] * round_factor) / round_factor
    #     tt.assert_equal(rounded_actual_tensor, rounded_expected_tensor)

if __name__ == '__main__':
    unittest.main()
