from .base import ContinualMethod
from data.replay import TaskBalancedBuffer


class Replay(ContinualMethod):
    uses_replay = True
    selection = "random"

    def __init__(self, config):
        super().__init__(config)
        self.buffer = TaskBalancedBuffer(config.get("replay", {}).get("capacity", 150), config["training"]["seed"])

    def after_task(self, model, current, loader, device):
        self.buffer.update(current, self.selection)
