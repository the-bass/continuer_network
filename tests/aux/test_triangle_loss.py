import unittest
import torch

from aux.triangle_loss import construct_triple, triple_area, TriangleLoss


class TestTriangleLoss(unittest.TestCase):

    def test_construct_triple(self):
        x = torch.autograd.Variable(torch.FloatTensor([
            [1, 2, 4, 5],
            [5, 6, 8, 9]
        ]))

        output = torch.autograd.Variable(torch.FloatTensor([
            3,
            7
        ]))

        self.assertTrue(torch.equal(
            construct_triple(x, output),
            torch.FloatTensor([
                [2, 3, 4],
                [6, 7, 8]
            ])
        ))

    def test_triple_area(self):
        # Area[(0, 1), (1, 0.5), (2, 3)] = 1/2 [(3 - 1) + 2 (1 - 0.5)] = 1/2 [2 + 1] = 1.5
        self.assertEqual(
            triple_area((1, 0.5, 3)),
            1.5
        )

        # Area[(0, 1), (1, 0.5), (2, 3)] = 1/2 [(25.6 - 16) + 2 (16 - 0.2)] = 1/2 [9.6 + 31.6] = 1/2 41.2 = 20.6
        self.assertEqual(
            triple_area((16, 0.2, 25.6)),
            20.6
        )

        self.assertTrue(torch.equal(
            triple_area(torch.FloatTensor([
                [1, 0.5, 3],
                [1, 1.5, 3],
                [16, 0.2, 25.6]
            ])),
            torch.FloatTensor([1.5, 0.5, 20.6])
        ))

    def test_triangle_loss(self):
        x = torch.autograd.Variable(torch.FloatTensor([
            [1, 1, 3, 5],
            [5, 16, 25.6, 9]
        ]))

        output = torch.autograd.Variable(torch.FloatTensor([
            0.5,
            0.2
        ]))

        y = torch.autograd.Variable(torch.FloatTensor([
            1.5,
            7
        ]))

        self.assertTrue(torch.equal(
            TriangleLoss(reduce=False)(x, output, y),
            torch.autograd.Variable(torch.FloatTensor([1, 6.8]))
        ))

        self.assertAlmostEqual(
            TriangleLoss()(x, output, y).data[0],
            3.9,
            places=6
        )

        self.assertAlmostEqual(
            TriangleLoss(size_average=False)(x, output, y).data[0],
            7.8,
            places=6
        )

if __name__ == '__main__':
    unittest.main()
