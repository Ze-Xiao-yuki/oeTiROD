import copy
import torch
from torch import nn
from memory.features import descriptor64


class SharedExpertDetector(nn.Module):
    """One shared backbone/FPN, independently adaptable detection heads."""
    def __init__(self, detector, num_experts=3, router_weight=1.0,
                 normalize=((103.53, 116.28, 123.675), (57.375, 57.12, 58.395))):
        super().__init__()
        if num_experts < 1:
            raise ValueError("At least one expert is required")
        self.backbone, self.fpn = detector.backbone, detector.fpn
        self.experts = nn.ModuleList([copy.deepcopy(detector.head) for _ in range(num_experts)])
        self.router = nn.Sequential(nn.Linear(66, 32), nn.ReLU(), nn.Linear(32, num_experts))
        self.register_buffer("image_mean", torch.tensor(normalize[0]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(normalize[1]).view(1, 3, 1, 1))
        self.selected_expert, self.router_weight = 0, router_weight
        self.last_experts = None
        self.shared_frozen = False

    @property
    def head(self):
        return self.experts[self.selected_expert]

    def route(self, images, features):
        rgb = (images.detach() * self.image_std + self.image_mean) / 255
        brightness = rgb.mean((1, 2, 3)).unsqueeze(1)
        contrast = rgb.std((1, 2, 3), unbiased=False).unsqueeze(1)
        summary = descriptor64(features[-1])
        return self.router(torch.cat([brightness, contrast, summary], dim=1))

    def select_training_expert(self, index, freeze_shared=True):
        if not 0 <= index < len(self.experts):
            raise ValueError(f"Expert index out of range: {index}")
        self.selected_expert, self.shared_frozen = index, freeze_shared
        for module in [self.backbone, self.fpn]:
            module.requires_grad_(not freeze_shared)
        for i, expert in enumerate(self.experts):
            expert.requires_grad_(i == index)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if self.shared_frozen:
                self.backbone.eval()
                self.fpn.eval()
            for i, expert in enumerate(self.experts):
                if i != self.selected_expert:
                    expert.eval()
        return self

    def forward(self, images):
        features = self.fpn(self.backbone(images))
        chosen = self.route(images, features).argmax(1)
        self.last_experts = chosen.detach()
        outputs = None
        for index in chosen.unique().tolist():
            mask = chosen == index
            prediction = self.experts[index]([x[mask] for x in features])
            if outputs is None:
                outputs = prediction.new_zeros((len(images), *prediction.shape[1:]))
            outputs[mask] = prediction
        return outputs

    def forward_train(self, batch):
        features = self.fpn(self.backbone(batch["img"]))
        predictions = self.head(features)
        detection_loss, states = self.head.loss(predictions, batch)
        targets = torch.as_tensor(batch.get("expert_id", [self.selected_expert] * len(batch["img"])),
                                  device=batch["img"].device, dtype=torch.long)
        router_loss = nn.functional.cross_entropy(self.route(batch["img"], features), targets)
        return predictions, detection_loss + self.router_weight * router_loss, dict(states, router=router_loss)

    def set_epoch(self, epoch):
        self.epoch = epoch
