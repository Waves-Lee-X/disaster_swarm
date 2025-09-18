from pathlib import Path
import numpy as np
import cv2
import time
from typing import List, Tuple

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dirs(*dirs):
    for d in dirs: Path(d).mkdir(parents=True, exist_ok=True)

def save_rgb(img_bgr: np.ndarray, path: Path):
    cv2.imwrite(str(path), img_bgr)

def save_depth_npy(depth: np.ndarray, path: Path):
    np.save(str(path), depth.astype(np.float32))

def save_depth_preview_png(depth: np.ndarray, path: Path):
    if depth.size == 0: return
    d = (depth - depth.min()) / max(1e-6, (depth.max() - depth.min()))
    cv2.imwrite(str(path), (d * 255).astype(np.uint8))

def save_yolo_annotation(path: Path, boxes: List[Tuple[int, float, float, float, float]], w: int, h: int):
    with open(path, "w", encoding="utf-8") as f:
        for cls, x1, y1, x2, y2 in boxes:
            x_c, y_c = ((x1 + x2) / 2.0) / w, ((y1 + y2) / 2.0) / h
            ww, hh = (x2 - x1) / w, (y2 - y1) / h
            f.write(f"{int(cls)} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}\n")
