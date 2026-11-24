from .rep import Replay
from memory.features import resnet_features


class KMeansReplay(Replay):
    def after_task(self, model, current, loader, device):
        features = resnet_features(current, self.config.get("resnet_weights"), device)
        self.buffer.update(current, "kmeans", features)
