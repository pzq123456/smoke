"""
告警管理器
- 冷却期控制：同一摄像头两次告警的最小间隔
- 连续帧确认：连续 N 帧检测到才触发告警（防止单帧误报）
- 帧标注：检测框 + 可选文字水印 → base64 编码
- 告警分发：调用 Webhook 等渠道推送
"""

import base64
import time
from datetime import datetime, timezone

import cv2
import simplejpeg
from loguru import logger

from server.core.detector import Detection
from server.alert.webhook import WebhookAlerter


class AlertManager:
    """管理单路摄像头的告警状态与分发。"""

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        target_classes: list[str] | None = None,
        save_frame_overlay: bool = False,
        cooldown_seconds: float = 30.0,
        min_detection_count: int = 3,
        debug_render: bool = False,
        webhook: WebhookAlerter | None = None,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头显示名称
            target_classes: 触发告警的目标类别列表，None 则匹配所有检测
            save_frame_overlay: 是否在证据帧上叠加摄像头名称/时间水印
            cooldown_seconds: 同一摄像头两次告警的最小间隔（秒）
            min_detection_count: 连续检测到抽烟的帧数阈值
            debug_render: True=详细标注（cigarette位置+置信度），False=简洁标注（仅人体框+SMOKING）
            webhook: Webhook 推送实例，None 则不推送
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.target_classes = target_classes
        self.save_frame_overlay = save_frame_overlay
        self.cooldown_seconds = cooldown_seconds
        self.min_detection_count = min_detection_count
        self.debug_render = debug_render
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
            # 按配置的目标类别过滤（None 表示匹配所有检测）
            if self.target_classes is not None:
                smoking_detections = [
                    d for d in detections if d.class_name in self.target_classes
                ]
            else:
                smoking_detections = detections

            if smoking_detections:
                self._consecutive_hits += 1
                for d in smoking_detections:
                    logger.debug("detect camera={} class={} conf={:.2f} hits={}/{}",
                                 self.camera_name, d.class_name, d.confidence,
                                 self._consecutive_hits, self.min_detection_count)

                if self._consecutive_hits >= self.min_detection_count:
                    if self._cooldown_passed():
                        self._trigger(frame, smoking_detections)
                        self._consecutive_hits = 0
                        return True
                    else:
                        # 冷却中，重置计数器避免冷却结束后立即再次触发
                        remain = self.cooldown_seconds - (time.time() - self._last_alert_time)
                        logger.debug("skip camera={} reason=cooldown remain={:.0f}s",
                                     self.camera_name, remain)
                        self._consecutive_hits = 0
            else:
                # 无检测 → 递减计数器（逐步递减，容忍偶尔丢帧）
                if self._consecutive_hits > 0:
                    self._consecutive_hits -= 1
                    logger.debug("hits camera={} {}→{}", self.camera_name,
                                 self._consecutive_hits + 1, self._consecutive_hits)

            return False
        except Exception:
            logger.exception("[{}] 告警处理异常，跳过本帧", self.camera_name)
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _cooldown_passed(self) -> bool:
        """检查冷却期是否已过。"""
        elapsed = time.time() - self._last_alert_time
        return elapsed >= self.cooldown_seconds

    def _trigger(self, frame, detections: list[Detection]):
        """触发告警：标注帧 → base64 编码 → 推送 Webhook。"""
        now = datetime.now(timezone.utc)
        iso_timestamp = now.isoformat()

        best_conf = max(d.confidence for d in detections)
        best_class = max(detections, key=lambda d: d.confidence).class_name
        logger.info("🚨 alert camera={} class={} conf={:.2f} hits={}/{}",
                    self.camera_name, best_class, best_conf,
                    self._consecutive_hits, self.min_detection_count)

        # 1. 标注检测框
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            # 人体框标签
            if self.debug_render:
                if d.person_confidence is not None:
                    label = f"SMOKING p{d.person_confidence:.2f}"
                else:
                    label = f"{d.class_name} {d.confidence:.2f}"
            else:
                label = "SMOKING"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # cigarette 子框（黄色，仅调试模式）
            if self.debug_render and d.sub_detections:
                for sub in d.sub_detections:
                    sx1, sy1, sx2, sy2 = sub.bbox
                    cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
                    cv2.putText(annotated, f"smoke {sub.confidence:.2f}",
                                (sx1, max(sy1 - 5, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 调试模式：叠加连续帧计数器
        if self.debug_render:
            cv2.putText(annotated,
                        f"hits: {self._consecutive_hits}/{self.min_detection_count}",
                        (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 2. 可选：叠加摄像头/时间水印
        if self.save_frame_overlay:
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

        # 3. 编码为 base64（simplejpeg SIMD 加速）
        buffer = simplejpeg.encode_jpeg(annotated, quality=85, colorspace='BGR')
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # 4. 构建告警 payload
        detections_payload = []
        for d in detections:
            entry = {
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": list(d.bbox),
            }
            if self.debug_render:
                entry["person_confidence"] = (
                    round(d.person_confidence, 3) if d.person_confidence is not None else None
                )
                entry["sub_detections"] = [
                    {
                        "class": sub.class_name,
                        "confidence": round(sub.confidence, 3),
                        "bbox": list(sub.bbox),
                    }
                    for sub in (d.sub_detections or [])
                ]
            detections_payload.append(entry)

        payload = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "timestamp": iso_timestamp,
            "detections": detections_payload,
            "frame_base64": frame_base64,
        }

        # 5. 推送 Webhook（失败不影响主流程）
        if self.webhook:
            try:
                self.webhook.send(payload)
            except Exception:
                logger.exception("Webhook 推送异常")

        self._last_alert_time = time.time()
