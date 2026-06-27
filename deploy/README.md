# 抽烟检测服务 — Docker 部署

## 目录结构

```
deploy/
├── README.md              ← 本文件
├── docker-compose.yml     ← 容器编排，一键启动/停止/重启
├── config.yaml            ← 服务配置（摄像头、webhook、模型路径等）
├── models/                ← 放模型权重文件（git 忽略）
│   └── best.pt
└── logs/                  ← 容器运行时日志（git 忽略）
```

## 快速开始

### 1. 放入模型

从训练产出中拷贝模型，或直接放在 `models/` 下：

```bash
# 从项目根目录执行
cp runs/detect/<你的训练目录>/weights/best.pt deploy/models/
```

### 2. 编辑配置

```bash
vim deploy/config.yaml
```

需要修改的内容：

| 配置项 | 说明 |
|--------|------|
| `model.path` | 模型在容器内的路径。拷贝到 models/ 则写 `/models/<文件名>`；直接引用训练产出则写 `/runs/detect/.../best.pt` |
| `model.target_classes` | 触发告警的目标类别名，必须与模型训练的类别名一致 |
| `cameras` | 摄像头列表，支持 rtsp 和 local 两种类型 |
| `alert.webhook.url` | 告警推送地址。测试时指向宿主机：`http://host.docker.internal:9999/api/smoke-alert` |
| `alert.cooldown_seconds` | 同摄像头两次告警的最小间隔 |
| `alert.min_detection_count` | 连续检测到目标的帧数阈值 |

### 3. 启动服务

```bash
cd deploy
docker compose up -d
```

### 4. 查看状态

```bash
# 查看容器状态
docker compose ps

# 查看实时日志
docker compose logs -f

# 宿主机上查看持久化日志
tail -f logs/server.log
```

### 5. 本地摄像头接入（Windows 测试用）

Docker Desktop 运行在 WSL2 虚拟机中，无法直接访问宿主机 USB 摄像头。
使用 `local/camera_stream.py` 将本地摄像头转为 HTTP MJPEG 流：

```bash
# 终端 1：启动推流（宿主机）
python local/camera_stream.py
# 或指定设备：python local/camera_stream.py --device 1
# 或用视频文件模拟：python local/camera_stream.py --video test.mp4
```

`deploy/config.yaml` 中摄像头配置：

```yaml
cameras:
  - id: "test"
    name: "本地测试"
    type: "rtsp"
    rtsp_url: "http://host.docker.internal:8080/stream"    # 指向宿主机推流
    enabled: true
```

> **原理**：容器内 `RTSPStreamer` 底层使用 `cv2.VideoCapture`，可无缝读取 HTTP MJPEG 流。
> 生产环境直接填 RTSP 摄像头地址即可，无需此脚本。

### 6. 测试告警

在宿主机另开终端，启动 webhook 接收器模拟消费端：

```bash
python local/webhook_receiver.py
```

告警触发后，证据帧会保存在 `alerts/` 目录下。

## 日常运维

```bash
# 所有操作在 deploy/ 目录下执行，或指定 compose 文件路径

# 换模型
cp <新模型> models/
vim config.yaml          # 改 model.path
docker compose restart

# 改配置
vim config.yaml
docker compose restart

# 改代码后重建镜像
docker compose build --no-cache
docker compose up -d

# 停止服务
docker compose down
```

## 模型路径参考

`config.yaml` 中 `model.path` 支持两种写法：

```yaml
# 方式 A：模型放在 deploy/models/ 下
model:
  path: "/models/best.pt"

# 方式 B：直接指向 runs/ 训练产出，免拷贝
model:
  path: "/runs/detect/yolo26m_smoking_20260626_0601/weights/best.pt"
```

两种方式在 `docker-compose.yml` 中都已配置好挂载，任意选择。

## 挂载关系

| 宿主机路径 | 容器内路径 | 读写 | 用途 |
|-----------|-----------|:----:|------|
| `deploy/config.yaml` | `/app/server/config.yaml` | 只读 | 配置文件 |
| `deploy/models/` | `/models/` | 只读 | 模型权重 |
| `../runs/`（项目根） | `/runs/` | 只读 | 训练产出，方便直接引用 |
| `deploy/logs/` | `/logs/` | 读写 | 服务日志持久化 |

## 宿主机工具使用说明

### 摄像头推流（Windows）

Windows 终端可能无法正确输出 emoji 字符，运行时建议设置编码：

```bash
# PowerShell / cmd
set PYTHONIOENCODING=utf-8 && uv run python local/camera_stream.py

# Git Bash
PYTHONIOENCODING=utf-8 uv run python local/camera_stream.py
```

### Webhook 接收器

```bash
PYTHONIOENCODING=utf-8 uv run python local/webhook_receiver.py
```

### 端到端测试流程

```bash
# 终端 1：启动摄像头推流（宿主机）
PYTHONIOENCODING=utf-8 uv run python local/camera_stream.py

# 终端 2：启动 Webhook 接收器（宿主机）
PYTHONIOENCODING=utf-8 uv run python local/webhook_receiver.py

# 终端 3：启动 Docker 检测服务
cd deploy && docker compose up -d

# 观察日志
docker compose logs -f

# 在摄像头前展示抽烟动作，观察告警是否触发
# 告警证据帧保存在 alerts/<camera_id>/ 目录下
# 持久化日志在 deploy/logs/server.log

# 测试完毕
docker compose down
# Ctrl+C 停止 camera_stream.py 和 webhook_receiver.py
```

## 已知问题与修复

### 1. PEP 668 — pip 安装被拒绝

基础镜像 `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime` 基于 Python 3.12+，
启用了 PEP 668 保护，直接 `pip install` 会报错：

```
error: externally-managed-environment
```

**修复**：Dockerfile 中 `pip install` 添加 `--break-system-packages` 参数（容器内无破坏系统 Python 的风险）。

### 2. ultralytics `half` 参数弃用

较新版本的 ultralytics 已弃用 `half` 参数，改用自动精度管理。
运行时会输出大量 `WARNING ⚠️ 'half' is deprecated` 日志。

**修复**：`server/core/detector.py` 中 `model.predict()` 调用已移除 `half` 参数。

## 架构扩展思路

当前架构核心模块边界清晰，常见扩展改动量不大：

### 配置来源切换（文件 → API）

配置加载入口集中在 `server/main.py` 的 `load_config()` 一个函数。改为启动时从 HTTP API 拉取配置，只需改这一处，其余代码不变。若需要运行时热更新配置，则额外增加一个后台轮询线程即可。

### 多 Webhook / 多通道输出

当前 `WebhookAlerter` 接口极简——只有一个 `send(payload) -> bool` 方法。扩展方式：

- **不同摄像头推不同 webhook**：摄像头配置中加 `webhook_url` 字段，为每路创建独立的 `WebhookAlerter` 注入 `AlertManager`
- **同时推多个通道**（如 webhook + MQTT + Kafka）：新增对应的 sender 类（实现 `send` 方法），在 `AlertManager._trigger()` 中遍历调用即可

关键文件索引：

| 文件 | 职责 |
|------|------|
| `server/main.py` | 配置加载 + 组件装配（唯一入口） |
| `server/alert/webhook.py` | Webhook 推送客户端（`send()` 接口） |
| `server/alert/manager.py` | 告警决策 + 帧编码 + 分发 |

---

## 对接说明

### Webhook 推送数据格式

检测到目标后，服务通过 HTTP POST 向配置的 webhook URL 推送 JSON，格式如下：

```json
{
  "camera_id": "gate",
  "camera_name": "工厂大门",
  "timestamp": "2026-06-27T09:30:00.123456+00:00",
  "detections": [
    {
      "class": "cigarette",
      "confidence": 0.89,
      "bbox": [320, 240, 400, 380]
    }
  ],
  "frame_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `camera_id` | string | 摄像头唯一标识，对应配置中的 `cameras[].id` |
| `camera_name` | string | 摄像头显示名称 |
| `timestamp` | string | ISO 8601 时间戳（UTC），触发告警的时刻 |
| `detections` | array | 本帧检测到的所有目标 |
| `detections[].class` | string | 目标类别名（与模型训练的类别名一致） |
| `detections[].confidence` | float | 置信度 0–1 |
| `detections[].bbox` | [int×4] | 边界框 `[x1, y1, x2, y2]`，像素坐标 |
| `frame_base64` | string | 标注后的证据帧 JPEG，base64 编码 |

**推送行为**：
- HTTP POST，Content-Type `application/json; charset=utf-8`
- 超时默认 10s，失败自动重试 2 次
- 同一摄像头受冷却期限制（默认 10s），冷却期内不重复推送
- 支持连续帧确认（默认 1 帧），防止单帧误报

接收端示例代码参见 `local/webhook_receiver.py`。

### 配置参数速查

#### model — 模型参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | **必填** | 模型权重文件在容器内的路径 |
| `conf` | float | `0.35` | 置信度阈值，低于此值的检测结果丢弃 |
| `device` | int/string | `0` | GPU 设备 ID；`"cpu"` 使用 CPU 推理 |
| `target_classes` | list | `["cigarette"]` | 触发告警的目标类别名，必须与模型训练时类别名一致 |

#### cameras — 摄像头列表

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | string | **必填** | 唯一标识，出现在告警 payload 的 `camera_id` 中 |
| `name` | string | **必填** | 显示名称，出现在告警 payload 的 `camera_name` 中 |
| `type` | string | `"rtsp"` | `"rtsp"`（网络流 / MJPEG）或 `"local"`（本地 USB 摄像头） |
| `rtsp_url` | string | type=rtsp 必填 | RTSP 或 HTTP MJPEG 流地址 |
| `device_id` | int | type=local 时 `0` | OpenCV 摄像头设备 ID |
| `enabled` | bool | `true` | 是否启用该路摄像头 |

#### alert — 告警控制

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cooldown_seconds` | float | `10` | 同一摄像头两次告警的最小间隔（秒），防止告警风暴 |
| `min_detection_count` | int | `1` | 连续检测到目标的帧数阈值，只有连续 N 帧都命中才触发 |
| `save_frame_overlay` | bool | `false` | 是否在证据帧上叠加摄像头名称/时间水印 |

#### alert.webhook — 推送目标

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | string | `null` | Webhook 接收地址；为 `null` 时不推送 |
| `timeout` | float | `10` | 单次请求超时（秒） |
| `retries` | int | `2` | 失败重试次数（不含首次） |

#### log — 日志

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `file` | string | `"/logs/server.log"` | 容器内日志路径（JSON 格式，10MB 轮转，保留 30 天） |

### 告警决策流程

```
每帧 → 检测到 target_classes？
         ↓ 是
      连续命中计数器 +1
         ↓
      达到 min_detection_count？
         ↓ 是
      距上次告警 > cooldown_seconds？
         ↓ 是
      🚨 标注帧 → JPEG base64 → Webhook POST
```

- **未命中时计数器递减**（非直接清零），容忍偶尔丢帧
- **计数器达到阈值后**：触发告警，计数器归零，更新冷却时间戳
- **冷却期内再命中**：计数器归零但不触发，避免冷却结束后立即再次告警
