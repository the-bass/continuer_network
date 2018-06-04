import torch.nn as nn


class SimpleRNN(nn.Module):

    def __init__(self, input_size=1):
        super().__init__()

        self.layer = nn.GRU(
            input_size=input_size,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, input):
        output, h_n = self.layer(input)

        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        return output
