"""
YOLO 模型封装
- 单例加载，所有摄像头 Worker 共享一个模型实例
- 提供结构化检测结果
"""

from pathlib import Path
from dataclasses import dataclass

from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """单条检测结果。"""
    class_name: str        # 类别名称，如 "smoking"
    confidence: float      # 置信度 0-1
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)


class SmokeDetector:
    """抽烟检测器，封装 YOLO 模型的加载与推理。"""

    def __init__(
        self,
        model_path: str | Path,
        conf: float | dict[str, float] = 0.35,
        device: int | str | None = 0,
    ):
        """
        Args:
            model_path: YOLO 模型权重路径
            conf: 置信度阈值。float 时全局统一；dict 时逐类设定（如 {'face': 0.5, 'smoking': 0.25}）。
                  YOLO 内部使用 min(dict.values()) 保证不漏检，后置逐类提纯。
            device: GPU 设备 ID（0, 1, ...），None 或 "cpu" 表示 CPU
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 多态 conf：dict 时取最低值给 YOLO 保召回，逐类过滤在 detect() 中完成
        if isinstance(conf, dict):
            self._class_conf = conf
            self._conf = min(conf.values())
        else:
            self._class_conf = None
            self._conf = conf

        logger.info("加载模型: {} (device={}, conf={})", model_path, device,
                    conf if self._class_conf is None else f"{self._conf}(yolo) / {self._class_conf}(per-class)")
        self._model = YOLO(str(model_path), task="detect")

        # 解析 device 参数
        if device is None or device == "cpu":
            self._device = "cpu"
        else:
            self._device = device

    def detect(self, frame) -> list[Detection]:
        """
        对单帧执行抽烟检测。

        Args:
            frame: BGR numpy 数组（来自 cv2）

        Returns:
            Detection 列表，未检测到任何目标时为空列表
        """
        results = self._model.predict(
            frame,
            conf=self._conf,
            verbose=False,
            device=self._device,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self._model.names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])

            # 逐类置信度后置过滤
            if self._class_conf is not None:
                min_conf = self._class_conf.get(class_name, self._conf)
                if confidence < min_conf:
                    continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections

    def annotate_frame(self, frame, detections: list[Detection]):
        """
        在帧上绘制检测框和标签（不修改原图，返回新图）。

        Args:
            frame: 原始帧
            detections: detect() 返回的检测列表

        Returns:
            标注后的帧
        """
        import cv2

        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        return annotated
