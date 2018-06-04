import torch.nn as nn
from torch_state_control.nn import StatefulModule


class Continuer(StatefulModule):

    def __init__(self):
        super().__init__("Continuer")

        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=20,
            num_layers=2,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=20, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        output, h_n = self.encoder(x)
        x = h_n[-1]

        x = self.decoder(x)

        return x
