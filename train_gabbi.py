import torch
import math
import coach
import coach.targets
from sequences_with_gaps_10_100_datasets import train_set, dev_set, test_set, set_size, amount_of_points
from network_interface import NetworkInterface
from networks.gabbi import Gabbi
from networks.gabbi2x import Gabbi2x


network = Gabbi2x()
shuffle = True
batch_size = 4096#8192
learning_rate = 0.1
cch = coach.Coach(network=network)

def train_one_epoch():
    network_interface = NetworkInterface(
        network=network,
        loss_class=torch.nn.MSELoss,
        batch_size=batch_size
    )
    epoch_loss = network_interface.train_one_epoch(
        train_set,
        shuffle=shuffle,
        learning_rate=learning_rate
    )

    return epoch_loss

def measure_performance():
    network_interface = NetworkInterface(
        network=network,
        loss_class=torch.nn.MSELoss,
        batch_size=batch_size
    )

    train_set_performance = network_interface.performance_on_set(train_set)
    dev_set_performance = network_interface.performance_on_set(dev_set)

    return train_set_performance, dev_set_performance

cch.train(
    target=coach.targets.infinity_reached(),
    # target=coach.targets.time_elapsed(30),
    # target=coach.targets.epoch_reached(2),
    train_one_epoch=train_one_epoch,
    measure_performance=measure_performance,
    checkpoint_frequency=3,
    checkpoint=-1,
    notes=f"lr={learning_rate}",
    # record=False
)
