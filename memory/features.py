"""Retrieval descriptors are never passed to a detection head as spatial features."""
import numpy as np


def kmeans_representatives(features, count, iterations=30):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite N x D matrix")
    count = min(count, len(x))
    if count <= 0:
        return []
    # Deterministic farthest-point initialization, followed by Lloyd updates.
    seeds = [0]
    while len(seeds) < count:
        distance = ((x[:, None] - x[seeds]) ** 2).sum(-1).min(1)
        distance[seeds] = -1
        seeds.append(int(distance.argmax()))
    centers = x[seeds].copy()
    for _ in range(iterations):
        labels = ((x[:, None] - centers) ** 2).sum(-1).argmin(1)
        updated = np.stack([x[labels == i].mean(0) if (labels == i).any() else centers[i] for i in range(count)])
        if np.allclose(updated, centers):
            break
        centers = updated
    chosen = []
    for center in centers:
        distance = ((x - center) ** 2).sum(1)
        distance[chosen] = np.inf
        chosen.append(int(distance.argmin()))
    return chosen


def descriptor64(feature):
    import torch
    from torch.nn import functional as F
    flat = feature.detach().float().flatten(1)
    descriptor = F.adaptive_avg_pool1d(flat.unsqueeze(1), 64).squeeze(1)
    return F.normalize(descriptor, dim=1)


def resnet_features(dataset, weights_path, device):
    """Explicit local weights prevent a hidden download or random feature encoder."""
    import torch
    from PIL import Image
    from torchvision import models, transforms
    if not weights_path:
        raise ValueError("KMREP requires resnet_weights pointing to local ResNet50 weights")
    encoder = models.resnet50(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.fc = torch.nn.Identity()
    encoder.to(device).eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    features = []
    with torch.no_grad():
        for item in dataset.data_info:
            from pathlib import Path
            with Image.open(Path(dataset.img_path) / item["file_name"]) as image:
                inputs = transform(image.convert("RGB")).unsqueeze(0).to(device)
            features.append(encoder(inputs).cpu().numpy()[0])
    return np.stack(features)
