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
