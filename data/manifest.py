"""COCO contract and split checks, independent of the detection runtime."""
import json
import math
from collections import defaultdict
from train_eval.config import split_paths


def validate_sequence(config, check_images=True):
    category_map = None
    image_sets = defaultdict(set)
    report = []
    for task in config["tasks"]:
        for split in ("train", "val", "test"):
            images, annotation = split_paths(config, task, split)
            with annotation.open(encoding="utf-8-sig") as stream:
                coco = json.load(stream)
            mapping = sorted((int(c["id"]), str(c["name"])) for c in coco["categories"])
            if not mapping or len({c[0] for c in mapping}) != len(mapping):
                raise ValueError(f"Invalid categories: {annotation}")
            if category_map is None:
                category_map = mapping
            if mapping != category_map:
                raise ValueError(f"Category mapping differs: {annotation}. Export a shared category table first.")
            records = coco["images"]
            ids = {i["id"] for i in records}
            if not records or len(ids) != len(records):
                raise ValueError(f"Empty images or duplicate image IDs: {annotation}")
            if len({a["id"] for a in coco["annotations"]}) != len(coco["annotations"]):
                raise ValueError(f"Duplicate annotation IDs: {annotation}")
            for item in records:
                image = (images / item["file_name"]).resolve()
                if image != images and images not in image.parents:
                    raise ValueError(f"Image outside its root: {image}")
                if check_images and not image.is_file():
                    raise FileNotFoundError(image)
                image_sets[split].add(image)
            categories = {c[0] for c in mapping}
            for box in coco["annotations"]:
                if box["image_id"] not in ids or box["category_id"] not in categories:
                    raise ValueError(f"Unresolved annotation reference: {annotation}")
                if len(box["bbox"]) != 4 or not all(math.isfinite(float(v)) for v in box["bbox"]) or box["bbox"][2] <= 0 or box["bbox"][3] <= 0:
                    raise ValueError(f"Invalid bounding box: {annotation}")
            report.append({"task": task["name"], "split": split, "images": len(records), "boxes": len(coco["annotations"])})
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = image_sets[left] & image_sets[right]
        if overlap:
            raise ValueError(f"Split leakage ({left}/{right}): {next(iter(overlap))}")
    return category_map, report
