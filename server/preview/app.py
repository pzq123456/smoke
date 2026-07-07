# server/preview/app.py

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger


def create_app(workers: dict) -> FastAPI:
    """创建预览 FastAPI 应用。workers 由 main.py 创建并传入。"""
    app = FastAPI(title="吸烟检测预览", version="0.3.0")
    app.state.workers = workers

    @app.get("/health")
    async def health(request: Request):
        """健康检查 — Docker / 负载均衡用."""
        statuses = {cid: w.is_running for cid, w in request.app.state.workers.items()}
        return {"ok": all(statuses.values()), "cameras": statuses}

    @app.get("/stream/{camera_id}")
    async def stream_mjpeg(camera_id: str, request: Request):
        """MJPEG 实时视频流."""
        w = request.app.state.workers.get(camera_id)
        if w is None:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

        async def generate():
            last_version = -1
            try:
                while w.is_running:
                    await w._frame_ready.wait()
                    w._frame_ready.clear()
                    if w._frame_version == last_version:
                        continue
                    last_version = w._frame_version
                    body = w._latest_jpeg_bytes
                    if body is None:
                        continue
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + body + b'\r\n')
            except asyncio.CancelledError:
                pass

        return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        loop = asyncio.get_running_loop()
        for w in app_instance.state.workers.values():
            w._loop = loop
            w.start()
        logger.info("预览已启动，{} 路摄像头", len(app_instance.state.workers))
        yield
        for w in app_instance.state.workers.values():
            w.stop()
        logger.info("预览已停止")

    app.router.lifespan_context = lifespan
    return app
