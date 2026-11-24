class ContinualMethod:
    uses_replay = False

    def __init__(self, config):
        self.config = config

    def before_task(self, model, task_index):
        pass

    def train_dataset(self, current):
        return current

    def prepare_mode(self, model):
        pass

    def loss(self, model, batch):
        _, loss, states = model.forward_train(batch)
        return loss, states

    def after_task(self, model, current, loader, device):
        pass

    def close(self):
        pass
