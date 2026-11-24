from .distillation import Distillation


class LearningWithoutForgetting(Distillation):
    """Match the previous detector's complete head output on current images."""
