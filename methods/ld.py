from .distillation import Distillation


class LatentDistillation(Distillation):
    intermediate = True
    freeze_backbone = True
