"""Bounded, task-balanced replay selection and finite mixed mini-batches."""
import math
import random


def temporal_indices(size, count):
    count = min(size, count)
    if count <= 0:
        return []
    if count == 1:
        return [size // 2]
    return [round(i * (size - 1) / (count - 1)) for i in range(count)]


class TaskBalancedBuffer:
    def __init__(self, capacity=150, seed=1234):
        if capacity < 0:
            raise ValueError("Buffer capacity must be nonnegative")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.groups = []

    def update(self, dataset, selection="random", features=None):
        if not len(dataset):
            raise ValueError("Cannot insert an empty task")
        all_groups = self.groups + [(dataset, list(range(len(dataset))))]
        quotas = [0] * len(all_groups)
        for _ in range(min(self.capacity, sum(len(indices) for _, indices in all_groups))):
            eligible = [i for i, (_, indices) in enumerate(all_groups) if quotas[i] < len(indices)]
            chosen = min(eligible, key=lambda i: (quotas[i], i))
            quotas[chosen] += 1
        updated = []
        for group_id, ((source, indices), quota) in enumerate(zip(all_groups, quotas)):
            if selection == "temporal":
                selected = [indices[i] for i in temporal_indices(len(indices), quota)]
            elif selection == "kmeans" and group_id == len(all_groups) - 1 and quota:
                from memory.features import kmeans_representatives
                selected = kmeans_representatives(features, quota)
            else:
                selected = self.rng.sample(indices, quota)
            updated.append((source, selected))
        self.groups = updated

    def __len__(self):
        return sum(len(indices) for _, indices in self.groups)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for source, indices in self.groups:
            if index < len(indices):
                return source[indices[index]]
            index -= len(indices)
        raise IndexError(index)


class MixedBatchSampler:
    """Every current sample is visited once; replay is drawn with replacement."""
    def __init__(self, current_size, replay_size, batch_size, seed=1234, replay_ratio=0.5):
        if current_size < 1 or batch_size < 1 or not 0 <= replay_ratio < 1:
            raise ValueError("Invalid mixed batch configuration")
        if replay_size and batch_size < 2 and replay_ratio:
            raise ValueError("Replay requires batch_size >= 2")
        self.current_size, self.replay_size, self.batch_size = current_size, replay_size, batch_size
        self.replay_count = min(batch_size - 1, max(1, round(batch_size * replay_ratio))) if replay_size and replay_ratio else 0
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return math.ceil(self.current_size / (self.batch_size - self.replay_count))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(self.current_size))
        rng.shuffle(indices)
        stride = self.batch_size - self.replay_count
        for start in range(0, len(indices), stride):
            batch = indices[start:start + stride]
            batch += [self.current_size + rng.randrange(self.replay_size) for _ in range(self.replay_count)]
            yield batch
