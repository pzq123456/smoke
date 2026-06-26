"""
告警管理器
- 冷却期控制：同一摄像头两次告警的最小间隔
- 连续帧确认：连续 N 帧检测到才触发告警（防止单帧误报）
- 关键帧保存：标注框和时间戳后保存为 JPG
- 告警分发：调用 Webhook 等渠道推送
"""

import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2

from server.core.detector import Detection
from server.alert.webhook import WebhookAlerter

logger = logging.getLogger("smoke_detector.alert")


class AlertManager:
    """管理单路摄像头的告警状态与分发。"""

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        cooldown_seconds: float = 30.0,
        min_detection_count: int = 3,
        save_dir: str | Path = "alerts",
        webhook: WebhookAlerter | None = None,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头显示名称
            cooldown_seconds: 同一摄像头两次告警的最小间隔（秒）
            min_detection_count: 连续检测到抽烟的帧数阈值
            save_dir: 关键帧保存根目录
            webhook: Webhook 推送实例，None 则不推送
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.cooldown_seconds = cooldown_seconds
        self.min_detection_count = min_detection_count
        self.save_dir = Path(save_dir) / camera_id
        self.webhook = webhook

        # 内部状态
        self._consecutive_hits = 0          # 连续检测到抽烟的帧数
        self._last_alert_time: float = 0.0  # 上一次告警的 Unix 时间戳

    # ------------------------------------------------------------------
    # 主入口：每帧调用
    # ------------------------------------------------------------------
    def handle(self, frame, detections: list[Detection]) -> bool:
        """
        处理一帧的检测结果，决定是否触发告警。

        Args:
            frame: 原始帧（未标注）
            detections: detect() 返回的检测列表

        Returns:
            True 表示触发了告警
        """
        try:
            smoking_detections = [d for d in detections if d.class_name == "smoking"]

            if smoking_detections:
                self._consecutive_hits += 1
                logger.debug(
                    "[%s] 检测到抽烟: %d 个目标, 连续 %d/%d 帧",
                    self.camera_name,
                    len(smoking_detections),
                    self._consecutive_hits,
                    self.min_detection_count,
                )

                if self._consecutive_hits >= self.min_detection_count:
                    if self._cooldown_passed():
                        self._trigger(frame, smoking_detections)
                        self._consecutive_hits = 0
                        return True
                    else:
                        # 冷却中，重置计数器避免冷却结束后立即再次触发
                        self._consecutive_hits = 0
            else:
                # 无检测 → 递减计数器（逐步递减，容忍偶尔丢帧）
                if self._consecutive_hits > 0:
                    self._consecutive_hits -= 1

            return False
        except Exception:
            logger.exception("[%s] 告警处理异常，跳过本帧", self.camera_name)
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _cooldown_passed(self) -> bool:
        """检查冷却期是否已过。"""
        elapsed = time.time() - self._last_alert_time
        return elapsed >= self.cooldown_seconds

    def _trigger(self, frame, detections: list[Detection]):
        """触发告警：保存帧 → 推送 Webhook。"""
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()

        logger.info(
            "🚨 告警触发 [%s] %d 个抽烟目标",
            self.camera_name, len(detections),
        )

        # 1. 标注并保存关键帧
        frame_path = self._save_frame(frame, detections, now, timestamp_str)

        # 2. 构建告警 payload
        payload = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "timestamp": iso_timestamp,
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "bbox": list(d.bbox),
                }
                for d in detections
            ],
            "frame_path": str(frame_path) if frame_path else None,
        }

        # 3. 推送 Webhook（失败不影响主流程）
        if self.webhook:
            try:
                self.webhook.send(payload)
            except Exception as e:
                logger.error("Webhook 推送异常: %s", e)

        self._last_alert_time = time.time()

    def _save_frame(
        self,
        frame,
        detections: list[Detection],
        now: datetime,
        timestamp_str: str,
    ) -> Path | None:
        """
        在帧上绘制标注框和时间戳并保存。
        返回保存路径，失败返回 None。
        """
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)

            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # 绘制检测框
            for d in detections:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{d.class_name} {d.confidence:.2f}"
                cv2.putText(
                    annotated, label,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            # 绘制时间戳和摄像头信息
            overlay_text = [
                f"Camera: {self.camera_name} ({self.camera_id})",
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            for i, text in enumerate(overlay_text):
                cv2.putText(
                    annotated, text,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

            file_path = self.save_dir / f"{timestamp_str}.jpg"
            cv2.imwrite(str(file_path), annotated)
            logger.info("关键帧已保存: %s", file_path)
            return file_path

        except Exception as e:
            logger.error("保存关键帧失败: %s", e)
            return None
