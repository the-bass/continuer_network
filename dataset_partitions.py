import dataset_parties
from datasets.sequences_dataset import SequencesDataset


dataset = SequencesDataset()

train_set, dev_set, test_set = dataset_parties.ng_style(dataset)
