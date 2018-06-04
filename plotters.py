import math
import matplotlib.pyplot as plt


def plot_loss_per_sample(performance_samples, show=True):
    losses = list(map(lambda x: x.loss, performance_samples))
    # y_axis = [cost for index, cost in self.indexed_and_sorted_costs]

    plt.title(f"Loss per sample")
    # plt.grid()

    conditional_plot_options = {
        'marker': '|'
    }
    if len(losses) > 1000:
        conditional_plot_options['marker'] = 'None'

    plt.plot(
        losses,
        color='#04151F',
        # label='Train set',
        linestyle='solid',
        markersize=4,
        **conditional_plot_options
    )

    plt.ylabel('Loss')
    plt.xlabel('Sample')

    if show:
        plt.show()

def plot_sample(performance_sample, name=None, show=True):
    loss = performance_sample.loss
    actual_y = performance_sample.actual_y
    predicted_y = performance_sample.predicted_y

    title = f"Loss: {loss:.4f}"

    if name:
        title = name + " - " + title

    plt.title(title)

    x = performance_sample.x.view(-1).numpy()
    #plt.suptitle(f"{name} - Length: {len(record[0])}")

    shared_plot_options = {
        'linestyle': 'solid',
        'marker': 'o',
        'markersize': 4
    }

    #graph_fractions, gaps = exm.split_at_gaps(record)

    plt.plot(range(x.shape[0]), list(x), color='green', linestyle='solid', marker='|', markersize=4)
    plt.plot([x.shape[0]], actual_y, color='green', label='Actual value', marker='|', markersize=8)
    plt.plot([x.shape[0]], predicted_y, color='red', label='Predicted value', marker='_', markersize=8)

    plt.legend(loc='best')

    if show:
        plt.show()

def plot_samples(performance_samples, cols=3):
    rows = math.ceil(len(performance_samples) / cols)

    plt.figure(num=0, figsize=(10, 2 * rows))

    for i, performance_sample in enumerate(performance_samples):
        sp = plt.subplot(rows, cols, i + 1)
        plot_sample(performance_sample, show=False)

    plt.tight_layout()
    plt.show()


""" AUX """

def gap_predicted(x, y):
    sequence = []

    for i in range(x.shape[0]):
        if x[i, 1] == 1:
            value = x[i, 0]
        elif x[i, 1] == 0:
            value = y[i]
        else:
            kkfmls

        sequence.append(value)

    return sequence
