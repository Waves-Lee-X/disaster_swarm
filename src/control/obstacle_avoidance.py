import numpy as np

class DepthAvoider:
    """基于 Depth 的快速避障：中心 ROI 小于阈值判障，尝试左右/前后偏移。"""
    def __init__(self, obstacle_th=2.0, roi_ratio=0.33, try_offsets=(3.0, -3.0, 0.0, 3.0)):
        self.obstacle_th = float(obstacle_th)
        self.roi_ratio = float(roi_ratio)
        # 尝试次序：右、左、前、后（你可按场景换顺序/幅度）
        self.try_offsets = try_offsets

    def obstacle_detected(self, depth):
        if depth is None or depth.size == 0: return False
        h, w = depth.shape
        rx = int(w * self.roi_ratio / 2)
        ry = int(h * self.roi_ratio / 2)
        cx, cy = w // 2, h // 2
        roi = depth[cy-ry:cy+ry, cx-rx:cx+rx]
        vals = roi[np.isfinite(roi)]
        if vals.size == 0: return False
        mean_d = float(np.mean(vals))
        return mean_d < self.obstacle_th
