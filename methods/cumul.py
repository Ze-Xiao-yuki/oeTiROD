from torch.utils.data import ConcatDataset
from .base import ContinualMethod


class Cumulative(ContinualMethod):
    def __init__(self, config):
        super().__init__(config)
        self.seen = []

    def train_dataset(self, current):
        self.seen.append(current)
        return ConcatDataset(self.seen)
