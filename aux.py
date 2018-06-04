import torch


def tensor_hash(tensor):
    """ Returns a hash representation of the given tensor. """
    return hash(tuples(tensor.tolist()))

def sample_hash(sample):
    """
    Returns a hash representation of the given sample. A sample is expected to
    have the form `(x, y)`, where x, y are `torch.tensor`s.
    """
    x, y = sample
    hashed_sample = hash(
        (tensor_hash(x), tensor_hash(y))
    )

    return hashed_sample

def hashed_dataset(dataset):
    return [sample_hash(sample) for sample in dataset]

def tuples(list_of_lists_of):
    """ Converts all lists (deep) inside the given list into tuples. """

    for index, element in enumerate(list_of_lists_of):
        if isinstance(list_of_lists_of[index], list):
            list_of_lists_of[index] = tuples(list_of_lists_of[index])

    return tuple(list_of_lists_of)

def meters_confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors. Meaning the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Dividing the 2 tensors returns a new tensor which holds a unique value
    # for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

    confusion = {
        'TP': true_positives,
        'FP': false_positives,
        'TN': true_negatives,
        'FN': false_negatives
    }

    return confusion

def from_confusion_string(confusion_string):
    tp, fp, tn, fn = confusion_string.split('|')

    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)

    return tp, fp, tn, fn

def to_confusion_string(confusion):
    return f"{confusion['TP']}|{confusion['FP']}|{confusion['TN']}|{confusion['FN']}"
