# 本地测试工具

宿主机（Windows）端辅助脚本，用于配合 Docker 检测服务进行本地测试。

## 工具列表

| 脚本 | 用途 |
|------|------|
| `camera_stream.py` | 将 Windows 本地摄像头转为 HTTP MJPEG 流（:8080/stream），供 Docker 容器读取 |
| `test_client.py` | 验证 MJPEG 推流是否正常（cv2.VideoCapture 拉流并显示画面） |
| `webhook_receiver.py` | 模拟第三方消费端，接收告警 POST，base64 解码存帧 + JSONL 日志 |
| `camera.py` | 独立 YOLO 检测测试（直接读摄像头 + 本地模型推理，不依赖 Docker） |

## 端到端测试（配合 Docker）

```
终端 1（推流）  →  PYTHONIOENCODING=utf-8 uv run python local/camera_stream.py
终端 2（接收）  →  PYTHONIOENCODING=utf-8 uv run python local/webhook_receiver.py
终端 3（检测）  →  cd deploy && docker compose up -d
```

数据流：

```
摄像头 → camera_stream.py (:8080 MJPEG)
       → Docker RTSPStreamer 读取
       → YOLO 推理
       → AlertManager 触发告警
       → Webhook POST (:9999)
       → webhook_receiver.py 存帧 → alerts/<camera_id>/
```

## 运行环境

项目使用 `uv` 管理 Python 依赖，所有脚本运行方式：

```bash
uv run python local/<脚本名>.py
```

### Windows 终端编码

Windows 终端（cmd/PowerShell）默认编码为 cp950/gbk，无法输出 emoji 字符。
`camera_stream.py` 的启动信息包含 emoji，运行时需设置：

```bash
# Git Bash
PYTHONIOENCODING=utf-8 uv run python local/camera_stream.py

# PowerShell
$env:PYTHONIOENCODING="utf-8"; uv run python local/camera_stream.py

# cmd
set PYTHONIOENCODING=utf-8 && uv run python local/camera_stream.py
```

## camera_stream.py

将本地摄像头推为 MJPEG HTTP 流。

```bash
PYTHONIOENCODING=utf-8 uv run python local/camera_stream.py [--device 0] [--port 8080] [--video test.mp4]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` / `-d` | `0` | OpenCV 摄像头设备 ID |
| `--video` / `-v` | — | 用视频文件代替摄像头（与 --device 互斥） |
| `--port` / `-p` | `8080` | HTTP 监听端口 |

```bash
# 默认摄像头
uv run python local/camera_stream.py

# 指定设备
uv run python local/camera_stream.py --device 1

# 用视频文件模拟
uv run python local/camera_stream.py --video test_video.mp4
```

Docker 容器中的 `deploy/config.yaml` 摄像头配置：

```yaml
cameras:
  - id: "test"
    name: "本地测试"
    type: "rtsp"
    rtsp_url: "http://host.docker.internal:8080/stream"
```

## test_client.py

验证推流是否正常——连接 MJPEG 流并用 OpenCV 显示画面。

```bash
uv run python local/test_client.py
```

按 `q` 或 `Esc` 退出。默认连接 `http://127.0.0.1:8080/stream`。

## webhook_receiver.py

模拟第三方消费端，接收检测服务推送的告警（含 base64 证据帧）。

```bash
PYTHONIOENCODING=utf-8 uv run python local/webhook_receiver.py [--port 9999] [--save-dir alerts]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` / `-p` | `9999` | 监听端口 |
| `--save-dir` | `alerts` | 证据帧保存目录 |

输出结构：

```
alerts/
├── payload.jsonl              # 结构化推送数据（JSONL，50MB 轮转）
└── <camera_id>/
    └── YYYYMMDD_HHMMSS.jpg    # 解码后的证据帧
```

告警 payload 格式参见 `server/README.md`。

## camera.py

独立测试脚本——直接用 YOLO 模型对摄像头画面推理并显示，不依赖 Docker。

```bash
uv run python local/camera.py
```

操作：
- `q` — 退出
- `s` — 保存当前帧
- `d` — 切换检测开关

模型路径默认指向 `runs/detect/yolo26m_smoking_20260626_0601/weights/best.pt`。
