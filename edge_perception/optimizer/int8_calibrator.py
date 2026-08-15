"""Entropy calibrator for INT8 TensorRT builds (Jetson-side).

Feed it 500-1000 frames sampled from real missions — calibration data must
match deployment distribution or INT8 accuracy will silently collapse.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # Jetson / TensorRT environments only
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    class ImageFolderCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, folder: str, imgsz: int = 640, batch: int = 8,
                     cache: str = "int8_calib.cache"):
            super().__init__()
            import cv2
            self.paths = sorted(Path(folder).glob("*.[jp][pn]g"))
            self.imgsz, self.batch, self.cache = imgsz, batch, cache
            self.idx = 0
            self.device_mem = cuda.mem_alloc(batch * 3 * imgsz * imgsz * 4)
            self._cv2 = cv2

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            if self.idx + self.batch > len(self.paths):
                return None
            imgs = []
            for p in self.paths[self.idx:self.idx + self.batch]:
                img = self._cv2.imread(str(p))
                img = self._cv2.resize(img, (self.imgsz, self.imgsz))
                img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                imgs.append(img)
            self.idx += self.batch
            batch = np.ascontiguousarray(np.stack(imgs))
            cuda.memcpy_htod(self.device_mem, batch)
            return [int(self.device_mem)]

        def read_calibration_cache(self):
            p = Path(self.cache)
            return p.read_bytes() if p.exists() else None

        def write_calibration_cache(self, cache):
            Path(self.cache).write_bytes(cache)

except Exception:  # pragma: no cover - laptop without TensorRT
    class ImageFolderCalibrator:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise RuntimeError("TensorRT + pycuda required (Jetson/JetPack).")
