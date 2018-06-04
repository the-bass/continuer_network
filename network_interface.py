import torch
import math
import sys


class PerformanceSample:

    def __init__(self, x, loss, actual_y, predicted_y):
        self.x = x
        self.loss = loss
        self.actual_y = actual_y
        self.predicted_y = predicted_y


class NetworkInterface:

    def __init__(self, network, loss_class=None, batch_size=None):
        self.network = network
        self.loss_class = loss_class if loss_class else torch.nn.MSELoss
        self.batch_size = batch_size if batch_size else 4096

    @staticmethod
    def __normalize_pair__(pair):
        x, y = pair

        max_in_pair = torch.max(x[:, 0])
        min_in_pair = torch.min(x[:, 0])

        offset = -min_in_pair
        factor = 1 / (max_in_pair + offset)
        normalized_x = (x + offset) * factor
        normalized_y = (y + offset) * factor
        normalized_pair = (normalized_x, normalized_y)

        return normalized_pair, factor, offset

    def __normalize_batch__(self, batch):
        """
        Normalize a batch (a list of samples) in the format

        [
            (torch.Tensor(m, n), torch.Tensor(u, v)),
            (torch.Tensor(s, t), torch.Tensor(x, y)),
            ...
        ]

        where m, n, u, v, s, t, x, y are natural numbers > 1.

        Returns the normalized batch in the same format.
        Returns the factors used to normalize each element in the batch.
        Returns the offests used to normalize each element in the batch.
        """

        normalized_batch = []
        factors = []
        offsets = []

        for pair in batch:
            normalized_pair, factor, offset = self.__normalize_pair__(pair)

            normalized_batch.append(normalized_pair)
            factors.append(factor)
            offsets.append(offset)

        return normalized_batch, torch.FloatTensor(factors), torch.FloatTensor(offsets)

    # @staticmethod
    # def __denormalize_tensor__(tensor, factors, offsets):
    #     """
    #     Denormalizes the given tensor (that represents a batch with the first
    #     dimension being the batch dimension) using the given factors and offsets.
    #
    #     For example, to normalize tensor[i], factors[i] and offsets[i] are being used
    #     to denormalize.
    #
    #     Returns tensor denormalized.
    #     """
    #
    #     assert(tensor.shape[0] == factors.shape[0])
    #     assert(tensor.shape[0] == offsets.shape[0])
    #
    #     def expand_for_element_wise_operation(factors, shape):
    #         hus = []
    #         for n in range(len(shape) - 1):
    #             hus.append(1)
    #         return factors.view(-1, *hus).expand(*shape)
    #
    #     denormalized_tensor = tensor / (expand_for_element_wise_operation(factors, tensor.shape) + sys.float_info.epsilon) - expand_for_element_wise_operation(offsets, tensor.shape)
    #
    #     return denormalized_tensor

    @staticmethod
    def __plugin_known_values__(y, x):
        """
        Fill in all values that are NOT gaps with the actual values from x.
        """

        unpacked_y, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            y,
            batch_first=True
        )
        unpacked_x, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            x,
            batch_first=True
        )

        for i in range(unpacked_y.shape[0]):
            for j in range(unpacked_y.shape[1]):
                if unpacked_x[i, j, 1] == 1:
                    unpacked_y[i, j, 0] = unpacked_x[i, j, 0]

        return torch.nn.utils.rnn.pack_padded_sequence(unpacked_y, lengths, batch_first=True)

    def __preprocessed_batch__(self, raw_batch, x_requires_grad=False):
        sorted_batch = sorted(raw_batch, key=lambda element: element[0].shape[0], reverse=True)

        if x_requires_grad:
            for x, y in sorted_batch:
                x.requires_grad = True
                # y.requires_grad = False

        # Normalize
        batch, normalization_factors, normalization_offsets = self.__normalize_batch__(sorted_batch)

        x = list(map(lambda element: element[0], batch))
        y = list(map(lambda element: element[1], batch))
        y = torch.cat(y)
        y = y.detach()

        # Pack 'n' pad
        x = torch.nn.utils.rnn.pack_sequence(x)
        # y = torch.nn.utils.rnn.pack_sequence(y)

        return x, y, normalization_factors, normalization_offsets

    def train_one_epoch(self, dataset, learning_rate, weight_decay, shuffle=True, timed=False):
        losses = []
        loss_function = self.loss_class()
        # optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay,
            amsgrad=False
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            collate_fn=lambda x: x
        )

        if timed:
            import time
            start = time.time()

        for batch in data_loader:
            optimizer.zero_grad()

            x, y, normalization_factors, normalization_offsets = self.__preprocessed_batch__(
                batch,
                x_requires_grad=True
            )

            if timed:
                end = time.time()
                print("Preprocessing:", end - start)
                start = time.time()

            predicted_y = self.network.forward(x)
            # predicted_y = self.__plugin_known_values__(predicted_y, x)

            if timed:
                end = time.time()
                print("Forward:", end - start)
                start = time.time()

            loss = loss_function(predicted_y, y)
            loss_float = loss.item() # NOTE: raises an exception if loss NOT 0-dimensional.

            if math.isnan(loss_float) or math.isinf(loss_float):
                raise ValueError(f"Loss is \"{loss_float}\". Exploding gradients?")

            losses.append(loss_float)

            if timed:
                end = time.time()
                print("Loss:", end - start)
                start = time.time()

            loss.backward()

            if timed:
                end = time.time()
                print("Backward:", end - start)
                start = time.time()

            optimizer.step() # Does the update

            if timed:
                end = time.time()
                print("Update:", end - start)
                start = time.time()

        mean_loss = torch.mean(torch.FloatTensor(losses))
        mean_loss_float = mean_loss.item()

        return mean_loss_float

    def performance_on_set(self, dataset):
        loss_function = self.loss_class()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x
        )

        loss_sum = 0
        losses = []
        for batch in data_loader:
            x, y, normalization_factors, normalization_offsets = self.__preprocessed_batch__(batch)

            predicted_y = self.network.forward(x)
            # predicted_y = self.__plugin_known_values__(predicted_y, x)

            loss = loss_function(predicted_y, y)
            loss_float = loss.item() # NOTE: raises an exception if loss NOT 0-dimensional.

            if math.isnan(loss_float) or math.isinf(loss_float):
                raise ValueError(f"Loss is \"{loss_float}\". Exploding gradients?")

            losses.append(loss_float)

        loss = torch.mean(torch.FloatTensor(losses))
        loss_float = loss.item()

        return loss_float

    def performance_on_samples(self, dataset):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x
        )

        loss_function = self.loss_class(reduce=False)
        predictions = []
        for batch in data_loader:
            x, y, normalization_factors, normalization_offsets = self.__preprocessed_batch__(batch)
            batch_size = len(batch)

            predicted_y = self.network.forward(x)
            # predicted_y = self.__plugin_known_values__(predicted_y, x)

            # unpacked_predicted_y, unpacked_predicted_y_lengths = torch.nn.utils.rnn.pad_packed_sequence(predicted_y, batch_first=True, padding_value=0)
            # unpacked_y, unpacked_y_lengths = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True, padding_value=0)
            unpacked_x, unpacked_x_lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

            loss = loss_function(predicted_y, y)
            # loss_float = loss.item() # NOTE: raises an exception if loss NOT 0-dimensional.

            # if math.isnan(loss_float) or math.isinf(loss_float):
            #     raise ValueError(f"Loss is \"{loss_float}\". Exploding gradients?")

            # print("unpacked_predicted_y", unpacked_predicted_y.shape)
            # print("unpacked_y", unpacked_y.shape)
            # print("unpacked_y_lengths", unpacked_y_lengths)
            # print("loss", loss.shape)
            # print("loss", loss.view(batch_size, -1).shape)
            # print("loss", torch.sum(loss.view(batch_size, -1), 1).shape)
            # print("per element loss", )

            # The third dimension is 1 and can be cut-off.
            # print("A", loss[0].item())
            # cleaned_loss = loss.view(batch_size, -1)

            # For each element in the batch, sum up the losses for all values of
            # the sequence.
            # summed_loss = torch.sum(cleaned_loss, 1)

            # Divide the loss sums through the lengths of the sequence for each
            # batch element.
            # loss_per_batch_element = torch.div(summed_loss, unpacked_y_lengths.float())

            for i, sequence_lengths in enumerate(unpacked_x_lengths):
                prediction = PerformanceSample(
                    x=unpacked_x[i, :sequence_lengths],
                    loss=loss[i].item(),
                    actual_y=y[i].item(),
                    predicted_y=predicted_y[i].item()
                )

                predictions.append(prediction)

        predictions.sort(key=lambda x: x.loss, reverse=True)

        return predictions
