"""PC/NX image or video inference. Writes results without opening a GUI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="task_XX/model.pt produced by run.py")
    parser.add_argument("--source", required=True, help="Image or video file; no desktop/camera capture")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=.35)
    args = parser.parse_args()
    import cv2
    import torch
    import yaml
    from train_eval.config import load_config, UniqueLoader
    from train_eval.runtime import activate_upstream, build_detector, prepare_batch
    config = load_config(args.config)
    activate_upstream()
    from nanodet.data.transform import Pipeline
    from nanodet.data.collate import naive_collate
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    categories = payload["categories"]
    detector = build_detector(config, len(categories))
    if payload.get("profile") == "project":
        from model.experts import SharedExpertDetector
        detector = SharedExpertDetector(detector, payload["experts"].get("count", 3))
    detector.load_state_dict(payload["model"], strict=True)
    detector.to(args.device).eval()
    version = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()[:16]
    with open(config["detector"], encoding="utf-8") as stream:
        preprocessing = yaml.load(stream, Loader=UniqueLoader)["data"]["val"]
    pipeline = Pipeline(preprocessing["pipeline"], preprocessing["keep_ratio"])
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    is_image = Path(args.source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    capture = None if is_image else cv2.VideoCapture(args.source)
    writer, frame_id = None, 0
    def frames():
        if is_image:
            # imdecode supports Unicode Windows paths.
            import numpy as np
            image = cv2.imdecode(np.fromfile(args.source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {args.source}")
            yield image
        else:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {args.source}")
            while True:
                valid, frame = capture.read()
                if not valid:
                    break
                yield frame
    try:
        with (destination / "detections.jsonl").open("w", encoding="utf-8") as output:
            for frame in frames():
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                start = time.perf_counter()
                height, width = frame.shape[:2]
                meta = {"img": frame.copy(), "raw_img": frame.copy(),
                        "img_info": {"id": frame_id, "height": height, "width": width, "file_name": Path(args.source).name}}
                meta = pipeline(None, meta, tuple(preprocessing["input_size"]))
                meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
                batch = prepare_batch(naive_collate([meta]), args.device)
                with torch.no_grad():
                    raw = detector(batch["img"])
                    result = detector.head.post_process(raw, batch)[frame_id]
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - start
                boxes = []
                for label, detections in result.items():
                    for box in detections:
                        if float(box[4]) < args.score:
                            continue
                        x1, y1, x2, y2 = map(int, box[:4])
                        category = categories[int(label)]
                        boxes.append({"category_id": category[0], "name": category[1], "xyxy": list(map(float, box[:4])), "score": float(box[4])})
                        color = ((int(label) * 67 + 53) % 200 + 30, (int(label) * 37 + 83) % 200 + 30, (int(label) * 97 + 23) % 200 + 30)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{category[1]} {box[4]:.2f}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                expert = int(detector.last_experts[0]) if hasattr(detector, "last_experts") else None
                output.write(json.dumps({"frame": frame_id, "model_version": version, "expert": expert,
                                         "preprocess_inference_postprocess_seconds": elapsed, "detections": boxes}, ensure_ascii=False) + "\n")
                if is_image:
                    cv2.imencode(".jpg", frame)[1].tofile(str(destination / "result.jpg"))
                else:
                    if writer is None:
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        writer = cv2.VideoWriter(str(destination / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 25, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create output video")
                    writer.write(frame)
                frame_id += 1
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
    print(f"Saved {frame_id} frames to {destination}")


if __name__ == "__main__":
    main()
