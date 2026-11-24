from .rep import Replay


class LatentReplay(Replay):
    """Preserve upstream LREP semantics: image replay with a frozen backbone.

    64-D retrieval descriptors are deliberately excluded from detection loss.
    """
    def before_task(self, model, task_index):
        self.frozen = task_index > 0
        model.backbone.requires_grad_(not self.frozen)

    def prepare_mode(self, model):
        if self.frozen:
            model.backbone.eval()
