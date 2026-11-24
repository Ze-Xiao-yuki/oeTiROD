import torch
from .base import ContinualMethod


class ElasticWeightConsolidation(ContinualMethod):
    def __init__(self, config):
        super().__init__(config)
        self.constraints = []

    def loss(self, model, batch):
        detection, states = super().loss(model, batch)
        penalty = detection.new_zeros(())
        for means, fisher in self.constraints:
            for name, parameter in model.named_parameters():
                if name in fisher:
                    penalty = penalty + (fisher[name].to(parameter.device) * (parameter - means[name].to(parameter.device)).square()).sum()
        total = detection + self.config.get("ewc", {}).get("strength", 10000) * penalty / 2
        return total, dict(states, ewc=penalty)

    def after_task(self, model, current, loader, device):
        from train_eval.runtime import prepare_batch
        selected = {name: p for name, p in model.named_parameters() if p.requires_grad and
                    any(key in name for key in ("backbone", "fpn", "head.cls_convs"))}
        fisher = {name: torch.zeros_like(p, device="cpu") for name, p in selected.items()}
        model.eval()
        count = 0
        for batch in loader:
            if count >= self.config.get("ewc", {}).get("fisher_batches", 32):
                break
            model.zero_grad(set_to_none=True)
            _, loss, _ = model.forward_train(prepare_batch(batch, device))
            loss.backward()
            for name, p in selected.items():
                if p.grad is not None:
                    fisher[name] += p.grad.detach().cpu().square()
            count += 1
        if not count:
            raise ValueError("No batches available for Fisher estimation")
        self.constraints.append(({name: p.detach().cpu().clone() for name, p in selected.items()},
                                 {name: value / count for name, value in fisher.items()}))
        model.zero_grad(set_to_none=True)
