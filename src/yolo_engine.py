import os
import cv2
import numpy as np
from typing import List, Dict
try:
    from . import config
except ImportError:
    import config

class YoloDetector:
    def __init__(self):
        self.backend = config.INFERENCE_BACKEND
        self.model = None
        self.class_names = None
        if self.backend == "roboflow":
            self._init_roboflow()
        else:
            self._init_ultralytics()

    def _init_ultralytics(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("未安装 ultralytics，请先 pip install ultralytics")
        if not os.path.exists(config.ULTRALYTICS_WEIGHTS):
            print(f"警告：未找到本地权重 {config.ULTRALYTICS_WEIGHTS}，请下载一个扑克牌检测模型的 .pt 文件并放置到该路径。")
        self.model = YOLO(config.ULTRALYTICS_WEIGHTS)
        self.class_names = self.model.names

    def _init_roboflow(self):
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("未检测到 ROBOFLOW_API_KEY，请在环境变量中配置后重试。")
        try:
            from roboflow import Roboflow
        except ImportError:
            raise RuntimeError("未安装 roboflow，请先 pip install roboflow")
        rf = Roboflow(api_key=api_key)
        ws = rf.workspace(config.ROBOFLOW_WORKSPACE)
        project = ws.project(config.ROBOFLOW_PROJECT)
        version = project.version(config.ROBOFLOW_VERSION)
        self.model = version.model  # 托管推理
        self.class_names = None

    def predict(self, frame_bgr) -> List[Dict]:
        if self.backend == "roboflow":
            return self._predict_roboflow(frame_bgr)
        else:
            return self._predict_ultralytics(frame_bgr)

    def _predict_ultralytics(self, frame_bgr):
        results = self.model.predict(source=frame_bgr, conf=config.ULTRA_CONF, verbose=False)
        dets = []
        if not results:
            return dets
        res = results[0]
        if not hasattr(res, "boxes") or res.boxes is None:
            return dets
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            cid = cls_ids[i]
            cls_name = self.class_names.get(cid, str(cid)) if isinstance(self.class_names, dict) else str(cid)
            conf = float(confs[i])
            dets.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "cls": cls_name,
                "conf": conf
            })
        return dets

    def _predict_roboflow(self, frame_bgr):
        pred = self.model.predict(frame_bgr, confidence=config.ROBOFLOW_CONF).json()
        dets = []
        for p in pred.get("predictions", []):
            cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cls_name = p.get("class", "card")
            conf = float(p.get("confidence", 0.0))
            dets.append({
                "box": [x1, y1, x2, y2],
                "cls": cls_name,
                "conf": conf
            })
        return dets