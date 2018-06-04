# import unittest
# import torch
#
# from aux.graph_data import costs_graph_data
#
#
# class TestGraphData(unittest.TestCase):
#
#     def test_costs_graph_data(self):
#         x = torch.FloatTensor([
#             [0.7],
#             [0.8]
#         ])
#
#         data = costs_graph_data(x)
#         self.assertEqual(data, [
#             (1, 0.800000011920929),
#             (0, 0.699999988079071)
#         ])
#
# if __name__ == '__main__':
#     unittest.main()
