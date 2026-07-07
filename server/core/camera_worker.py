"""
单路摄像头 Worker
- 每个摄像头一个独立线程
- 主循环：读取帧 → 检测 → 告警处理 →（可选）JPEG 编码发布
- 定期输出状态日志
"""

import asyncio
import time
import threading

import cv2
import simplejpeg
from loguru import logger

from server.core.detector import SmokeDetector, Detection
from server.alert.manager import AlertManager


class CameraWorker:
    """负责一路摄像头的检测与告警，可选输出 MJPEG 预览帧。"""

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        streamer,           # RTSPStreamer | LocalStreamer — 任何有 read()/stop()/connected 的对象
        detector: SmokeDetector,
        alert_manager: AlertManager,
        jpeg_quality: int | None = None,
        status_interval: int = 100,
        summary_interval: float = 60.0,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头名称（用于日志和告警）
            streamer: 视频流读取器（RTSPStreamer / LocalStreamer），由外部创建
            detector: 共享的 SmokeDetector 实例
            alert_manager: 该摄像头的 AlertManager 实例
            jpeg_quality: JPEG 质量 1-100，None 表示不编码（headless 模式）
            status_interval: 每隔多少帧打印一次 DEBUG 状态日志（默认 100）
            summary_interval: 每隔多少秒打印一次 INFO 摘要（默认 60）
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._streamer = streamer
        self.detector = detector
        self.alert_manager = alert_manager
        self._jpeg_quality = jpeg_quality
        self.status_interval = status_interval
        self.summary_interval = summary_interval

        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

        # ── 预览 Pub/Sub（仅 jpeg_quality 非 None 时启用）──
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fps: float = 0.0
        if jpeg_quality is not None:
            self._latest_jpeg_bytes: bytes | None = None
            self._frame_version: int = 0
            self._frame_ready = asyncio.Event()
        else:
            self._latest_jpeg_bytes = None
            self._frame_version = 0
            self._frame_ready = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """启动 Worker 线程。

        preview 模式下需先注入 _loop（由 FastAPI lifespan 在 uvicorn 事件循环中调用），
        headless 模式下 _loop 为 None 亦可正常运行。
        """
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"worker-{self.camera_id}"
        )
        self._thread.start()
        logger.debug("[{}] Worker 线程已启动", self.camera_name)

    def stop(self):
        """停止 Worker 线程并释放资源。"""
        self._stopped.set()
        # 唤醒可能正在等待的 HTTP 协程
        if self._frame_ready is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._frame_ready.set)
            except Exception:
                pass
        if self._streamer:
            self._streamer.stop()
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("[{}] Worker 已停止", self.camera_name)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _run(self):
        """Worker 主循环。"""
        frame_count = 0
        alert_count = 0
        t_start = time.time()
        t_last_summary = t_start

        logger.debug("[{}] 开始检测", self.camera_name)

        while not self._stopped.is_set():
            try:
                # 1. 读取帧
                t1 = time.time()
                frame = self._streamer.read()
                if frame is None:
                    if not self._streamer.connected:
                        self._stopped.wait(1)
                    continue

                # 2. 检测
                try:
                    detections = self.detector.detect(frame)
                except Exception as e:
                    logger.error("[{}] 检测异常: {}", self.camera_name, e)
                    continue

                t2 = time.time()
                now = time.time()

                # 3. 告警处理（防御：handle 内部异常不杀死 Worker；每帧调用以支持连续帧计数）
                try:
                    if self.alert_manager.handle(frame, detections):
                        alert_count += 1
                except Exception:
                    logger.exception("[{}] 告警处理异常", self.camera_name)

                # 4. 可选：JPEG 编码 + 无锁发布（仅 preview 模式）
                if self._jpeg_quality is not None:
                    self._annotate(frame, detections)
                    buf = simplejpeg.encode_jpeg(
                        frame, quality=self._jpeg_quality, colorspace='BGR')
                    self._latest_jpeg_bytes = buf
                    self._frame_version += 1
                    if self._loop is not None:
                        self._loop.call_soon_threadsafe(self._frame_ready.set)

                # 5. 统计
                frame_count += 1

                # -- 更新 FPS（两个统计块共用） --
                now_elapsed = now - t_start
                if frame_count % self.status_interval == 0 or now - t_last_summary >= self.summary_interval:
                    self._fps = frame_count / now_elapsed if now_elapsed > 0 else 0

                # -- DEBUG: 逐帧统计（仅 DEBUG 级别可见） --
                if frame_count % self.status_interval == 0:
                    inference_ms = (t2 - t1) * 1000
                    logger.debug(
                        "[{}] 帧: {} | 推理: {:.1f}ms | FPS: {:.1f} | 运行: {:.0f}s",
                        self.camera_name, frame_count, inference_ms, self._fps, now_elapsed,
                    )

                # -- DEBUG: 时间摘要（默认每 60 秒一条） --
                if now - t_last_summary >= self.summary_interval:
                    logger.debug("status camera={} fps={:.1f} frames={} alerts={} uptime={:.0f}s",
                                 self.camera_name, self._fps, frame_count, alert_count, now_elapsed)
                    t_last_summary = now

            except Exception:
                logger.exception(
                    "[{}] Worker 线程未预期异常，跳过本帧继续运行", self.camera_name
                )
                time.sleep(0.1)

        # 退出统计
        elapsed = time.time() - t_start
        logger.info("[{}] stop frames={} alerts={} uptime={:.0f}s fps={:.1f}",
                    self.camera_name, frame_count, alert_count, elapsed,
                    frame_count / elapsed if elapsed > 0 else 0)

    # ------------------------------------------------------------------
    # 帧标注（preview 模式用）
    # ------------------------------------------------------------------
    def _annotate(self, frame, detections):
        """标注检测框（原地绘制，仅 bbox + label，无水印）。"""
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = (0, 255, 0) if d.class_name == 'person' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if d.sub_detections:
                for s in d.sub_detections:
                    cv2.rectangle(frame, (s.bbox[0], s.bbox[1]),
                                  (s.bbox[2], s.bbox[3]), (0, 255, 255), 1)
                    cv2.putText(frame, f"{s.class_name} {s.confidence:.2f}",
                                (s.bbox[0], max(s.bbox[1] - 5, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
