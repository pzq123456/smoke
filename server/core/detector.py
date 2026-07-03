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
class SubDetection:
    """子检测结果（如 cigarette 在人体 ROI 内的位置，坐标为全帧坐标）。"""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) 全局坐标


@dataclass
class Detection:
    """单条检测结果。"""
    class_name: str        # 类别名称，如 "smoking"
    confidence: float      # 置信度 0-1
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    person_confidence: float | None = None       # 第一阶段 person 置信度（仅两阶段模式）
    sub_detections: list[SubDetection] | None = None  # cigarette 子检测列表（仅两阶段模式）


class SmokeDetector:
    """抽烟检测器，封装 YOLO 模型的加载与推理。

    支持两种模式：
    - 单阶段（默认）：直接在全帧上检测抽烟目标
    - 两阶段（配置 person_model 时启用）：先检测人体，再在人体 ROI 内检测抽烟，显著降低误报
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float | dict[str, float] = 0.35,
        device: int | str | None = 0,
        person_model_path: str | Path | None = None,
        person_conf: float = 0.4,
        imgsz: int | None = None,
    ):
        """
        Args:
            model_path: YOLO 抽烟模型权重路径
            conf: 置信度阈值。float 时全局统一；dict 时逐类设定（如 {'face': 0.5, 'smoking': 0.25}）。
                  YOLO 内部使用 min(dict.values()) 保证不漏检，后置逐类提纯。
            device: GPU 设备 ID（0, 1, ...），None 或 "cpu" 表示 CPU
            person_model_path: 可选的人体检测模型路径（COCO 预训练，class 0=person）。
                               配置后启用两阶段检测：先找人 → 再在人体 ROI 内检测抽烟。
            person_conf: 人体检测置信度阈值（默认 0.4）
            imgsz: 抽烟模型输入尺寸（默认 None，走 YOLO 内部默认值 640）
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

        logger.info("加载抽烟模型: {} (device={}, conf={})", model_path, device,
                    conf if self._class_conf is None else f"{self._conf}(yolo) / {self._class_conf}(per-class)")
        self._model = YOLO(str(model_path), task="detect")

        # 解析 device 参数
        if device is None or device == "cpu":
            self._device = "cpu"
        else:
            self._device = device

        # --- 可选：人体检测模型（第一阶段） ---
        self._person_model = None
        self._person_conf = person_conf
        self._imgsz = imgsz
        if person_model_path is not None:
            person_model_path = Path(person_model_path)
            if not person_model_path.exists():
                raise FileNotFoundError(f"人体检测模型不存在: {person_model_path}")
            logger.info("加载人体检测模型: {} (conf={})", person_model_path, person_conf)
            self._person_model = YOLO(str(person_model_path), task="detect")

    def detect(self, frame) -> list[Detection]:
        """
        对单帧执行抽烟检测。

        若配置了人体检测模型则走两阶段管线：
        1. 检测人体（COCO class 0）
        2. 在每个人体 ROI 内检测抽烟目标
        未配置人体模型时走单阶段全帧检测。

        Args:
            frame: BGR numpy 数组（来自 cv2）

        Returns:
            Detection 列表，未检测到任何目标时为空列表
        """
        if self._person_model is not None:
            return self._detect_two_stage(frame)

        # --- 单阶段模式：全帧检测 ---
        results = self._model.predict(
            frame,
            conf=self._conf,
            imgsz=self._imgsz,
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

    # ------------------------------------------------------------------
    # 两阶段检测（人体 → ROI → 抽烟）
    # ------------------------------------------------------------------
    def _detect_two_stage(self, frame) -> list[Detection]:
        """
        两阶段检测管线：
        1. 人体检测：用 COCO 预训练模型检测 frame 中所有人（class 0）
        2. ROI 抽烟检测：裁剪每个人体区域，用抽烟模型在该 ROI 内检测
        3. 坐标映射：将 ROI 内坐标映射回全帧坐标

        Returns:
            Detection 列表（class_name 为模型类别名，bbox 为全帧坐标）
        """
        h, w = frame.shape[:2]
        detections: list[Detection] = []

        # --- 第一阶段：人体检测 ---
        person_results = self._person_model.predict(
            frame,
            conf=self._person_conf,
            classes=[0],          # COCO class 0 = person
            verbose=False,
            device=self._device,
        )

        if not person_results or person_results[0].boxes is None:
            return detections

        boxes_data = person_results[0].boxes.data.cpu().numpy()
        if len(boxes_data) == 0:
            return detections

        # --- 第二阶段：逐人体 ROI 检测抽烟 ---
        for box in boxes_data:
            x1, y1, x2, y2 = map(int, box[:4])
            p_conf = float(box[4])  # person 置信度
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 跳过过小的 ROI（避免无效推理）
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # 对 ROI 执行抽烟检测
            try:
                smoke_results = self._model.predict(
                    person_roi,
                    conf=self._conf,
                    imgsz=self._imgsz,
                    verbose=False,
                    device=self._device,
                )
            except Exception as e:
                logger.warning(
                    "人体 ROI [{},{} -> {},{}] 抽烟检测异常: {}",
                    x1, y1, x2, y2, e,
                )
                continue

            if not smoke_results or smoke_results[0].boxes is None:
                continue

            # 收集该 ROI 内所有有效抽烟检测，取最高置信度
            best_conf = 0.0
            best_class = None
            sub_dets: list[SubDetection] = []
            for smoke_box in smoke_results[0].boxes:
                cls_id = int(smoke_box.cls[0])
                class_name = self._model.names.get(cls_id, str(cls_id))
                confidence = float(smoke_box.conf[0])

                # 逐类置信度后置过滤（与单阶段逻辑一致）
                if self._class_conf is not None:
                    min_conf = self._class_conf.get(class_name, self._conf)
                    if confidence < min_conf:
                        continue

                # 子检测坐标映射到全局
                sx1, sy1, sx2, sy2 = smoke_box.xyxy[0].tolist()
                gx1, gy1 = x1 + int(sx1), y1 + int(sy1)
                gx2, gy2 = x1 + int(sx2), y1 + int(sy2)
                sub_dets.append(SubDetection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(gx1, gy1, gx2, gy2),
                ))

                if confidence > best_conf:
                    best_conf = confidence
                    best_class = class_name

            # 若该人体区域检测到抽烟，输出人体框 + 子检测列表
            if best_class is not None:
                detections.append(Detection(
                    class_name=best_class,
                    confidence=best_conf,
                    bbox=(x1, y1, x2, y2),
                    person_confidence=p_conf,
                    sub_detections=sub_dets,
                ))

        return detections

