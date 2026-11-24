"""STM candidates, LTM annotated exemplars, PM immutable release snapshots."""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil


@dataclass
class MemoryRecord:
    key: str
    image: str
    task: str
    scene: str
    timestamp: str
    labels: list
    annotated: bool = False
    quality: float = 1.0
    descriptor: list = None
    encoder_version: str = ""


class HierarchicalMemory:
    def __init__(self, root, stm_capacity=128, ltm_capacity=150):
        if stm_capacity < 1 or ltm_capacity < 0:
            raise ValueError("Invalid memory capacity")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stm = deque(maxlen=stm_capacity)
        self.ltm_capacity = ltm_capacity
        self.ltm = {}
        index = self.root / "ltm.json"
        if index.exists():
            self.ltm = {x["key"]: MemoryRecord(**x) for x in json.loads(index.read_text(encoding="utf-8"))}

    def observe(self, record):
        self.stm.append(record)

    def promote(self, min_quality=0.0, max_similarity=0.999):
        import numpy as np
        for record in self.stm:
            if not record.annotated or record.quality < min_quality:
                continue
            if record.descriptor:
                feature = np.asarray(record.descriptor, dtype=float)
                if feature.shape != (64,) or not np.isfinite(feature).all():
                    raise ValueError("Retrieval descriptor must have 64 finite values")
                duplicate = False
                for old in self.ltm.values():
                    if old.descriptor and old.encoder_version == record.encoder_version and old.scene == record.scene and old.labels == record.labels:
                        previous = np.asarray(old.descriptor)
                        similarity = float(feature @ previous / max(np.linalg.norm(feature) * np.linalg.norm(previous), 1e-12))
                        if similarity >= max_similarity:
                            duplicate = True
                            break
                if duplicate:
                    continue
            self.ltm[record.key] = record
        # Greedy coverage allocation prioritizes rare scene/category combinations.
        candidates = list(self.ltm.values())
        selected, scene_counts, class_counts = [], Counter(), Counter()
        while candidates and len(selected) < self.ltm_capacity:
            def priority(r):
                coverage = 1 / (1 + scene_counts[r.scene])
                coverage += sum(1 / (1 + class_counts[c]) for c in r.labels) / max(1, len(r.labels))
                return coverage, r.quality, r.key
            chosen = max(candidates, key=priority)
            candidates.remove(chosen)
            selected.append(chosen)
            scene_counts[chosen.scene] += 1
            class_counts.update(chosen.labels)
        self.ltm = {r.key: r for r in selected}
        self.stm.clear()
        self.save()

    def save(self):
        destination = self.root / "ltm.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps([asdict(r) for r in self.ltm.values()], ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)

    def archive(self, checkpoint, version, categories, sample_index=None):
        if Path(version).name != version or any(c in version for c in '/\\:') or version in (".", ".."):
            raise ValueError("Invalid archive version")
        destination = self.root / "pm" / version
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / "model.pt"
        shutil.copy2(checkpoint, target)
        metadata = {"version": version, "categories": categories,
                    "samples": sample_index if sample_index is not None else list(self.ltm),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        (destination / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination
