"""OCR from supplied pixels only; no case/annotation access."""
from importlib.util import find_spec


class RapidOCRBackend:
    available = find_spec("rapidocr_onnxruntime") is not None

    def __init__(self):
        self._engine = None

    def read(self, image):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        import numpy as np
        # RapidOCR expects OpenCV BGR pixels; no file paths are supplied.
        rows, _ = self._engine(np.array(image)[:, :, ::-1].copy())
        return [{"polygon": [[float(x), float(y)] for x, y in box],
                 "text": text, "confidence": float(confidence)}
                for box, text, confidence in (rows or [])]
