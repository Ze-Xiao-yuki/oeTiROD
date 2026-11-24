import csv
import json
import math
from pathlib import Path


class TaskMatrix:
    def __init__(self, task_names, metric="mAP"):
        self.names, self.metric = list(task_names), metric
        self.values = [[None] * len(self.names) for _ in self.names]

    def set(self, trained, tested, score):
        if not 0 <= tested <= trained < len(self.names) or not math.isfinite(float(score)):
            raise ValueError("Invalid lower-triangular matrix entry")
        self.values[trained][tested] = float(score)

    def summary(self, row):
        scores = self.values[row][:row + 1]
        if any(value is None for value in scores):
            raise ValueError("Incomplete task evaluation row")
        forgetting = []
        for task in range(row):
            history = [self.values[stage][task] for stage in range(task, row)]
            if any(value is None for value in history):
                raise ValueError("Incomplete historical evaluation column")
            forgetting.append(max(history) - self.values[row][task])
        # Negative forgetting means backward improvement and is retained.
        return {"average_performance": sum(scores) / len(scores),
                "forgetting": sum(forgetting) / len(forgetting) if forgetting else 0.0}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "task_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["trained/tested", *self.names])
            writer.writerows([[name, *["" if v is None else v for v in row]] for name, row in zip(self.names, self.values)])
        summaries = {self.names[i]: self.summary(i) for i, row in enumerate(self.values) if all(v is not None for v in row[:i + 1])}
        (directory / "metrics.json").write_text(json.dumps({"metric": self.metric, "rows": summaries}, indent=2), encoding="utf-8")
