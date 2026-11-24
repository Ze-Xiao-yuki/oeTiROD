import argparse
import json
from pathlib import Path
from methods.registry import METHODS
from train_eval.config import load_config, split_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="TiROD 持续目标检测：十种方法与报告项目模块")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tirod.yml"))
    parser.add_argument("--method", type=str.upper, choices=list(METHODS), default="FT")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 / cpu; PC与NX共用代码入口")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="显示计划；不读取图像、不训练、不写入实验目录")
    parser.add_argument("--check-data", action="store_true", help="检查COCO、类别表、图像和划分重叠后退出")
    arguments = parser.parse_args(argv)
    if arguments.list_methods:
        for name, (_, _, source, description) in METHODS.items():
            print(f"{name:6s} {description} （原入口：{source}）")
        return
    config = load_config(arguments.config)
    if arguments.dry_run:
        tasks = []
        for task in config["tasks"]:
            tasks.append({"name": task["name"], "source": task.get("source"), "expert": task.get("expert", 0),
                          "splits": {split: {"images": str(split_paths(config, task, split)[0]),
                                             "annotations": str(split_paths(config, task, split)[1]),
                                             "annotations_exist": split_paths(config, task, split)[1].is_file()} for split in ("train", "val", "test")}})
        print(json.dumps({"method": arguments.method, "profile": config.get("profile", "baseline"), "device": arguments.device,
                          "output": config["output"], "training": config["training"], "tasks": tasks}, ensure_ascii=False, indent=2))
        return
    if arguments.check_data:
        from data.manifest import validate_sequence
        categories, report = validate_sequence(config)
        print(json.dumps({"categories": categories, "splits": report}, ensure_ascii=False, indent=2))
        return
    from train_eval.engine import run
    print(run(config, arguments.method, arguments.device))
