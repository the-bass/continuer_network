import torch.nn as nn


class SimpleFC(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(
            in_features=4,
            out_features=1,
            bias=True
        )

    def forward(self, input):
        return self.layer(input)
