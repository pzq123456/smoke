#!/usr/bin/env python3
import argparse
from contextlib import asynccontextmanager
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

parser = argparse.ArgumentParser(description="MJPEG 推流服务")
src = parser.add_mutually_exclusive_group()
src.add_argument("--device", "-d", type=int, default=0, help="摄像头 ID")
src.add_argument("--video", "-v", type=str, help="视频文件路径")
parser.add_argument("--port", "-p", type=int, default=8080, help="端口")
args = parser.parse_args()

source = args.video if args.video else args.device
cap = cv2.VideoCapture()

# 使用现代的 lifespan 管理生命周期，消除 DeprecationWarning
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：打开摄像头
    print(f"正在尝试打开外部设备/文件: {source} ...")
    cap.open(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"❌ 警告：无法打开源 {source}，请检查设备号或文件路径！")
    else:
        print("✅ 摄像头/视频源已成功初始化。")
    yield
    # 关闭时：释放资源
    cap.release()
    print("停止推流并释放资源。")

app = FastAPI(lifespan=lifespan)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.get("/")
def index():
    return {"status": "OK", "stream_url": "/stream"}

@app.get("/stream")
def stream():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print(f"📷 容器配置路径 -> http://host.docker.internal:{args.port}/stream")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")