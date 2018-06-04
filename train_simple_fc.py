import torch
import pandas
import os

import constants
from networks.simple_fc import SimpleFC
from datasets.one_hole_train import OneHoleTrain
from datasets.one_hole_dev import OneHoleDev
import coach
import coach.targets


train_dataset = OneHoleTrain(
    embedding_layers=7,
    currency='Euro',
    disjunct=True
)

dev_dataset = OneHoleDev(
    embedding_layers=7,
    currency='Euro',
    disjunct=True
)

print(f"Number of examples in train set: {len(train_dataset)}")
print(f"Number of examples in dev set: {len(dev_dataset)}")

shuffle = True
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=shuffle,
    drop_last=shuffle
)
network = SimpleFC()
optimizer = torch.optim.SGD(network.parameters(), lr=0.005, momentum=0.9)
cch = coach.Coach(network=network)

def performance_on_set(data_set):
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=128,
        shuffle=False,
        drop_last=False
    )

    loss_sum = 0
    for batch_idx, sample_batched in enumerate(data_loader):
        x = torch.autograd.Variable(sample_batched['x'], requires_grad=False)
        y = torch.autograd.Variable(sample_batched['y'].float(), requires_grad=False)

        output = network.forward(x)

        loss_sum += torch.nn.MSELoss(size_average=False)(output, y)

    loss = loss_sum.data[0] / len(data_set)

    return loss

def train_one_epoch():
    # count = 0s
    for batch_idx, sample_batched in enumerate(train_loader):
        # if count > 1:
        #     break
        # count += 1
        x = torch.autograd.Variable(sample_batched['x'], requires_grad=True)
        y = torch.autograd.Variable(sample_batched['y'].float(), requires_grad=False)
        # print('x', x)
        optimizer.zero_grad() # zero the gradient buffers
        output = network.forward(x)
        # print("output", output)
        # print("y", y)
        loss = torch.nn.MSELoss()(output, y)
        # print("loss", loss)
        loss.backward()
        optimizer.step() # Does the update

    epoch_loss = loss.data[0]
    # print("epoch_loss", epoch_loss)

    return epoch_loss

def measure_performance():
    train_set_performance = performance_on_set(train_dataset)
    dev_set_performance = performance_on_set(dev_dataset)

    return train_set_performance, dev_set_performance

cch.train(
    target=coach.targets.time_elapsed(15)
    train_one_epoch=train_one_epoch,
    measure_performance=measure_performance,
    checkpoint_frequency=2,
    checkpoint=-1,
    notes="one_hole embedding_layers=7 currency=Euro disjunct=True batch_size=128 lr=0.005 momentum=0.9",
    # record=False
)
