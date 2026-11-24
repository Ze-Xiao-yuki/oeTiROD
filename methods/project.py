"""Report prototype: hierarchical exemplar replay, separate from baselines."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from .base import ContinualMethod
from memory.hierarchy import HierarchicalMemory, MemoryRecord
from data.replay import temporal_indices


class RecordReplay:
    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return len(self.owner.memory.ltm)

    def __getitem__(self, index):
        key = list(self.owner.memory.ltm)[index]
        dataset, position = self.owner.lookup[key]
        return dataset[position]


class ProjectReplay(ContinualMethod):
    uses_replay = True

    def __init__(self, config):
        super().__init__(config)
        settings = config.get("memory", {})
        self.memory = HierarchicalMemory(Path(config["run_dir"]) / "memory", settings.get("stm_capacity", 128), settings.get("ltm_capacity", 150))
        self.lookup = {}
        self.buffer = RecordReplay(self)
        self.task_index = 0

    def before_task(self, model, task_index):
        self.task_index = task_index

    def after_task(self, model, current, loader, device):
        import torch
        from nanodet.data.collate import naive_collate
        from train_eval.runtime import prepare_batch
        from memory.features import descriptor64
        settings = self.config.get("memory", {})
        count = settings.get("candidates_per_task", 256)
        task = self.config["tasks"][self.task_index]
        # Training views contain random augmentation; descriptors use a stable view.
        import copy
        import yaml
        from nanodet.data.transform import Pipeline
        from train_eval.config import UniqueLoader
        with open(self.config["detector"], encoding="utf-8") as stream:
            preprocessing = yaml.load(stream, Loader=UniqueLoader)["data"]["val"]
        descriptor_dataset = copy.copy(current.dataset)
        descriptor_dataset.pipeline = Pipeline(preprocessing["pipeline"], preprocessing["keep_ratio"])
        descriptor_dataset.input_size = preprocessing["input_size"]
        encoder_hash = hashlib.sha256()
        for module in (model.backbone, model.fpn):
            for name, value in module.state_dict().items():
                encoder_hash.update(name.encode())
                encoder_hash.update(value.detach().cpu().numpy().tobytes())
        encoder_version = encoder_hash.hexdigest()[:16]
        model.eval()
        for position in temporal_indices(len(current), count):
            info = current.data_info[position]
            image = Path(current.img_path) / info["file_name"]
            key = hashlib.sha256(image.read_bytes()).hexdigest()
            annotation = current.coco_api.loadAnns(current.coco_api.getAnnIds(imgIds=[info["id"]]))
            labels = sorted({a["category_id"] for a in annotation})
            meta = descriptor_dataset.get_val_data(position)
            with torch.no_grad():
                batch = prepare_batch(naive_collate([meta]), device)
                feature = model.fpn(model.backbone(batch["img"]))[-1]
                descriptor = descriptor64(feature).cpu().tolist()[0]
            self.lookup[key] = (current, position)
            self.memory.observe(MemoryRecord(key, str(image), task["name"], task.get("scene", task["name"]),
                                            datetime.now(timezone.utc).isoformat(), labels, True, 1.0,
                                            descriptor, encoder_version))
            if len(self.memory.stm) == self.memory.stm.maxlen:
                self.memory.promote()
        self.memory.promote()
