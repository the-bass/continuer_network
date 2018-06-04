import torch
import pandas
import os
import sys

import constants
from networks.simple_fc import SimpleFC
from networks.norm_fc import NormFC
from datasets.one_hole_train import OneHoleTrain
from datasets.one_hole_dev import OneHoleDev
import coach
import coach.targets


train_dataset = OneHoleTrain(
    embedding_layers=10,
    disjunct=True
)

dev_dataset = OneHoleDev(
    embedding_layers=10,
    disjunct=True
)

print(f"Number of examples in train set: {len(train_dataset)}")
print(f"Number of examples in dev set: {len(dev_dataset)}")

shuffle = True
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=shuffle,
    drop_last=shuffle
)
network = NormFC()
optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
cch = coach.Coach(network=network)

def normalize(x):
    medians = torch.median(x, dim=-1)[0].view(-1, 1)
    expanded_medians = medians.expand(-1, x.shape[1])
    max_offs = torch.max(torch.abs(x - expanded_medians), -1)[0].view(-1, 1)
    expanded_max_offs = max_offs.expand(-1, x.shape[1]) + sys.float_info.epsilon
    x = (x - expanded_medians) / expanded_max_offs + 1

    return x, medians, max_offs

def denormalize(x, medians, max_offs):
    return x * max_offs + medians

def performance_on_set(data_set):
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=512,
        shuffle=False,
        drop_last=False
    )

    loss_sum = 0
    for batch_idx, sample_batched in enumerate(data_loader):
        x = torch.autograd.Variable(sample_batched['x'], requires_grad=False)
        y = torch.autograd.Variable(sample_batched['y'].float(), requires_grad=False)

        x_norm, medians, max_offs = normalize(x)
        y_norm, medians, max_offs = normalize(y.view(-1, 1))
        output_norm = network.forward(x_norm)
        loss_sum += torch.nn.MSELoss(size_average=False)(output_norm, y_norm)

    loss = loss_sum.data[0] / len(data_set)

    return loss

def train_one_epoch():
    # count = 0
    losses = []
    for batch_idx, sample_batched in enumerate(train_loader):
        # if count > 0:
        #     break
        # count += 1
        x = torch.autograd.Variable(sample_batched['x'], requires_grad=True)
        y = torch.autograd.Variable(sample_batched['y'].float(), requires_grad=False)
        # print('x', x)
        optimizer.zero_grad() # zero the gradient buffers
        x_norm, medians, max_offs = normalize(x)
        # print("x", x)
        # print("x_norm", x_norm)
        output_norm = network.forward(x_norm)
        y_norm, medians, max_offs = normalize(y.view(-1, 1))
        loss = torch.nn.MSELoss()(output_norm, y_norm)
        losses.append(loss.data[0])
        # print("loss", loss)
        loss.backward()
        optimizer.step() # Does the update

    # print("losses", torch.FloatTensor(losses))
    epoch_loss = torch.mean(torch.FloatTensor(losses))

    # print("epoch_loss", epoch_loss)

    return epoch_loss

def measure_performance():
    train_set_performance = performance_on_set(train_dataset)
    dev_set_performance = performance_on_set(dev_dataset)

    return train_set_performance, dev_set_performance

cch.train(
    target=coach.targets.time_elapsed(15),
    # target=coach.targets.epoch_reached(1),
    train_one_epoch=train_one_epoch,
    measure_performance=measure_performance,
    checkpoint_frequency=2,
    checkpoint=-1,
    notes="one_hole embedding_layers=10 disjunct=False batch_size=512 lr=0.001 momentum=0.9",
    # record=False
)
