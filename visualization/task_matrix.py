"""Render an exported task matrix; blank future cells are masked."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--output", default="task_matrix.png")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import numpy as np
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    names = rows[0][1:]
    matrix = np.array([[float(value) if value else np.nan for value in row[1:]] for row in rows[1:]])
    figure, axis = plt.subplots(figsize=(max(6, len(names) * .8), max(5, len(names) * .7)))
    plot = axis.imshow(np.ma.masked_invalid(matrix), cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=40, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Evaluated task")
    axis.set_ylabel("Trained through task")
    for row, column in np.argwhere(np.isfinite(matrix)):
        axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    figure.colorbar(plot, ax=axis, label="mAP (0–1)")
    figure.tight_layout()
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
