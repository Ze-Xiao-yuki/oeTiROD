from .base import ContinualMethod


class FineTuning(ContinualMethod):
    """Only the current task; parameters carry forward to the next task."""
