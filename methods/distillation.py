import copy
import torch
from torch.nn import functional as F
from .base import ContinualMethod


class Distillation(ContinualMethod):
    intermediate = False
    freeze_backbone = False

    def __init__(self, config):
        super().__init__(config)
        self.teacher = None

    def before_task(self, model, task_index):
        self.teacher = copy.deepcopy(model).eval().requires_grad_(False) if task_index else None
        model.backbone.requires_grad_(not (self.freeze_backbone and task_index > 0))

    def prepare_mode(self, model):
        if self.teacher is not None:
            self.teacher.eval()
            if self.freeze_backbone:
                model.backbone.eval()

    def loss(self, model, batch):
        if self.teacher is None:
            return super().loss(model, batch)
        student_features, teacher_features, handles = [], [], []
        if self.intermediate:
            for network, collection in [(model, student_features), (self.teacher, teacher_features)]:
                for tower in network.head.cls_convs:
                    handles.append(tower[-1].register_forward_hook(lambda module, args, output, collection=collection: collection.append(output)))
        try:
            predictions, loss, states = model.forward_train(batch)
            with torch.no_grad():
                old_predictions = self.teacher(batch["img"])
            if self.intermediate:
                count = model.head.num_classes
                distill = F.mse_loss(predictions[..., :count], old_predictions[..., :count])
                hidden = sum(F.mse_loss(current, old) for current, old in zip(student_features, teacher_features))
            else:
                distill = F.mse_loss(predictions, old_predictions)
                hidden = loss.new_zeros(())
            settings = self.config.get("distillation", {})
            total = loss + settings.get("alpha", 1.0) * distill + settings.get("beta", 1.0) * hidden
            return total, dict(states, distillation=distill, intermediate=hidden)
        finally:
            for handle in handles:
                handle.remove()

    def close(self):
        self.teacher = None
