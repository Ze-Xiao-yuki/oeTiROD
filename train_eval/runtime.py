"""Small bridge to the vendored NanoDet detection engine; no training on import."""
import sys
from pathlib import Path


def activate_upstream():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.append(root)


def prepare_batch(batch, device):
    import torch
    images = batch["img"]
    if isinstance(images, list):
        # Local equivalent of NanoDet stack_batch_img, zero padding to stride 32.
        height = (max(image.shape[-2] for image in images) + 31) // 32 * 32
        width = (max(image.shape[-1] for image in images) + 31) // 32 * 32
        tensor = images[0].new_zeros((len(images), images[0].shape[0], height, width), device=device)
        for index, image in enumerate(images):
            tensor[index, :, :image.shape[-2], :image.shape[-1]] = image.to(device)
        batch["img"] = tensor
    else:
        batch["img"] = images.to(device)
    return batch


def build_detector(config, num_classes):
    import yaml
    from train_eval.config import UniqueLoader
    activate_upstream()
    from nanodet.model.arch import build_model
    from nanodet.util.yacs import CfgNode
    with open(config["detector"], encoding="utf-8") as stream:
        detector = yaml.load(stream, Loader=UniqueLoader)
    detector["model"]["arch"]["head"]["num_classes"] = num_classes
    return build_model(CfgNode(detector["model"]))


class TaggedDataset:
    def __init__(self, dataset, expert):
        self.dataset, self.expert = dataset, expert

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        record["expert_id"] = self.expert
        return record

    @property
    def data_info(self):
        return self.dataset.data_info

    @property
    def img_path(self):
        return self.dataset.img_path

    @property
    def coco_api(self):
        return self.dataset.coco_api


def build_dataset(config, task, split, num_classes):
    from train_eval.config import detector_config
    activate_upstream()
    from nanodet.data.dataset import build_dataset as factory
    from nanodet.util.yacs import CfgNode
    _, source = detector_config(config, task, split, num_classes)
    dataset = factory(CfgNode(source), mode="train" if split == "train" else "val")
    return TaggedDataset(dataset, task.get("expert", 0)) if split == "train" else dataset


def dataloader(dataset, config, shuffle=False, sampler=None):
    import torch
    activate_upstream()
    from nanodet.data.collate import naive_collate
    options = dict(num_workers=config["training"]["workers"], collate_fn=naive_collate,
                   pin_memory=torch.cuda.is_available())
    if sampler is not None:
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **options)
    return torch.utils.data.DataLoader(dataset, batch_size=config["training"]["batch_size"],
                                       shuffle=shuffle, drop_last=False, **options)


def evaluate(model, dataset, config, device, output):
    import json
    import torch
    activate_upstream()
    from nanodet.evaluator.coco_detection import CocoDetectionEvaluator
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    predictions = {}
    with torch.no_grad():
        for batch in dataloader(dataset, config):
            batch = prepare_batch(batch, device)
            raw = model(batch["img"])
            predictions.update(model.head.post_process(raw, batch))
    evaluator = CocoDetectionEvaluator(dataset)
    # Upstream returns only a dict on empty predictions, (dict, table) otherwise.
    result = evaluator.evaluate(predictions, str(output))
    metrics, table = result if isinstance(result, tuple) else (result, "")
    metrics = {key: float(value) for key, value in metrics.items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "per_class.txt").write_text(table, encoding="utf-8")
    return metrics
