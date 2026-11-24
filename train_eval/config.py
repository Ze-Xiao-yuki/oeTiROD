import copy
from pathlib import Path
import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.load(stream, Loader=UniqueLoader)
    if not isinstance(config, dict) or not config.get("tasks"):
        raise ValueError("Configuration must contain a nonempty tasks list")
    names = [str(t["name"]) for t in config["tasks"]]
    if len(set(names)) != len(names):
        raise ValueError("Task names must be unique")
    for name in names:
        if not name or Path(name).name != name or name in (".", "..") or any(c in name for c in '/\\:'):
            raise ValueError(f"Unsafe task name: {name}")
    root = Path(config.get("data_root", ".")).expanduser()
    config["data_root"] = str((path.parent / root).resolve())
    model_path = Path(config.get("detector", "nanodet.yml"))
    config["detector"] = str((path.parent / model_path).resolve())
    output = Path(config.get("output", "../outputs"))
    config["output"] = str((path.parent / output).resolve())
    for field in ("initial_checkpoint", "resnet_weights"):
        if config.get(field):
            config[field] = str((path.parent / config[field]).resolve())
    training = config.setdefault("training", {})
    for key, default in [("epochs", 50), ("batch_size", 8), ("workers", 0), ("seed", 1234), ("val_interval", 1)]:
        training.setdefault(key, default)
    if any(training[k] < 1 for k in ["epochs", "batch_size", "val_interval"]) or training["workers"] < 0:
        raise ValueError("Invalid training epochs/batch_size/val_interval/workers")
    if config.get("profile", "baseline") not in ("baseline", "project"):
        raise ValueError("profile must be baseline or project")
    return config


def split_paths(config, task, split):
    spec = task[split]
    root = Path(config["data_root"])
    return (root / spec["images"]).resolve(), (root / spec["annotations"]).resolve()


def detector_config(config, task, split, num_classes):
    with open(config["detector"], encoding="utf-8") as stream:
        result = yaml.load(stream, Loader=UniqueLoader)
    result = copy.deepcopy(result)
    result["model"]["arch"]["head"]["num_classes"] = num_classes
    images, annotations = split_paths(config, task, split)
    source = result["data"]["train" if split == "train" else "val"]
    source.update(name="CocoDataset", img_path=str(images), ann_path=str(annotations))
    return result, source
