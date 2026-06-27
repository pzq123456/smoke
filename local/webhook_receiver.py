#!/usr/bin/env python3
"""
Webhook 测试接收器
- 启动一个 HTTP 服务器，接收抽烟检测服务端推送的告警
- 零依赖，纯 Python 标准库
- 格式化打印收到的告警内容

启动方式:
    python local/webhook_receiver.py                  # 默认 0.0.0.0:9999
    python local/webhook_receiver.py --port 8888      # 自定义端口
    python local/webhook_receiver.py --save alerts.json  # 保存到文件
"""

import argparse
import json
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
class WebhookHandler(BaseHTTPRequestHandler):
    """接收 POST 请求，解析 JSON body 并打印。"""

    # 类变量：累计接收计数 & 可选保存列表
    receive_count = 0
    saved_alerts: list | None = None
    save_path: str | None = None

    def log_message(self, format, *args):
        """接管日志，使用自定义格式。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {args[0]}", flush=True)

    def do_POST(self):
        """处理 POST 请求。"""
        # 读取 body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        # 尝试解析 JSON
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON", "raw": body.decode("utf-8", errors="replace")})
            return

        # 计数 & 可选保存
        WebhookHandler.receive_count += 1
        count = WebhookHandler.receive_count
        if WebhookHandler.saved_alerts is not None:
            WebhookHandler.saved_alerts.append({
                "received_at": datetime.now().isoformat(),
                "source": self.client_address[0],
                "payload": payload,
            })

        # 格式化打印
        self._print_alert(count, payload)

        self._respond(200, {"status": "ok", "received": count})

    def do_GET(self):
        """GET 请求：简单的健康检查 + 统计。"""
        info = {
            "service": "Webhook Test Receiver",
            "received_count": WebhookHandler.receive_count,
            "save_path": WebhookHandler.save_path,
        }
        self._respond(200, info)

    # ------------------------------------------------------------------
    def _respond(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _print_alert(self, count: int, payload: dict):
        """格式化打印告警内容。"""
        sep = "-" * 60
        source_ip = self.client_address[0]
        detections = payload.get("detections", [])
        timestamp = payload.get("timestamp", "N/A")

        print(f"\n{sep}")
        print(f" Alert #{count}  <- {source_ip}")
        print(f"{sep}")
        print(f" Camera:   {payload.get('camera_name', '?')} ({payload.get('camera_id', '?')})")
        print(f" Time:     {timestamp}")
        print(f" Objects:  {len(detections)}")
        for i, d in enumerate(detections[:10]):  # 最多显示 10 个
            bbox = d.get("bbox", [])
            print(f"   [{i+1}] {d.get('class','?')} conf={d.get('confidence',0):.2f} "
                  f"bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
        if len(detections) > 10:
            print(f"   ... and {len(detections) - 10} more")
        frame = payload.get("frame_path")
        if frame:
            print(f" Frame:    {frame}")
        print(f"{sep}\n", flush=True)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Webhook 测试接收器")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", "-p", type=int, default=9999, help="监听端口（默认 9999）")
    parser.add_argument("--save", "-s", default=None, help="保存收到的告警到 JSON 文件")
    args = parser.parse_args()

    # 可选：初始化保存列表
    if args.save:
        WebhookHandler.saved_alerts = []
        WebhookHandler.save_path = args.save

    # 启动服务器
    server = HTTPServer((args.host, args.port), WebhookHandler)
    print("=" * 60)
    print(" Webhook Test Receiver Started")
    print(f" Listening on: http://{args.host}:{args.port}")
    print(" All POST requests will be received and printed")
    if args.save:
        print(f" Save to file: {args.save}")
    print(" Press Ctrl+C to stop")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")

        # 保存（如果有）
        if WebhookHandler.saved_alerts is not None and args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(WebhookHandler.saved_alerts, f, ensure_ascii=False, indent=2, default=str)
            print(f"Saved {len(WebhookHandler.saved_alerts)} alerts to: {save_path}")

        server.shutdown()
        print("Receiver stopped")


if __name__ == "__main__":
    main()
