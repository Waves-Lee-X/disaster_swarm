import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class Detector:
    def __init__(self, model_path: str = None, device: str = "cpu", conf: float = 0.25):
        self.model = None
        self.conf = float(conf)
        if model_path and YOLO is not None:
            self.model = YOLO(model_path)
            self.device = device
        else:
            self.model = None

    def available(self) -> bool:
        return self.model is not None

    def detect(self, img_bgr: np.ndarray):
        if not self.available():
            return []  # 无模型时返回空结果
        # ultralytics 接受 RGB
        img_rgb = img_bgr[:, :, ::-1]
        results = self.model.predict(img_rgb, imgsz=640, conf=self.conf, device=self.device, verbose=False)
        out = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None: continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cls = int(b.cls.item())
                conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
                out.append((cls, x1, y1, x2, y2, conf))
        return out

