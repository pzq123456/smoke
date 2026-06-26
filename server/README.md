# 抽烟检测服务端

基于 YOLO 的抽烟行为实时检测后台服务。支持多路 RTSP / 本地摄像头接入，检测到抽烟后通过 Webhook 推送告警并保存关键帧证据。

## 目录结构

```
server/
├── main.py                 # 入口
├── config.yaml             # 配置文件（修改此处即可，无需改代码）
├── core/
│   ├── streamer.py         # 视频流读取器（RTSP + 本地摄像头）
│   ├── detector.py         # YOLO 模型封装
│   └── camera_worker.py    # 单路摄像头 Worker 线程
├── alert/
│   ├── webhook.py          # Webhook HTTP POST 推送
│   └── manager.py          # 告警管理（冷却 + 连续帧确认 + 关键帧保存）
├── utils/
│   └── logger.py           # 结构化日志
└── README.md
```

## 快速开始

```bash
# 使用默认配置启动
python -m server.main

# 指定配置文件
python -m server.main --config my_config.yaml
```

## 配置文件参考 (config.yaml)

### model — 模型参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | **必填** | YOLO 模型权重路径，相对于项目根目录 |
| `conf` | float | `0.35` | 置信度阈值，低于此值的检测结果被丢弃 |
| `device` | int/string/null | `0` | GPU 设备 ID；`0`=第一块GPU；`null`或`"cpu"`=CPU推理 |

### cameras — 摄像头列表

每路摄像头是一个数组元素：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | string | **必填** | 唯一标识，用于告警 payload 和帧保存目录 |
| `name` | string | **必填** | 显示名称，用于日志和告警 |
| `type` | string | `"rtsp"` | 摄像头类型：`rtsp`（RTSP网络流）或 `local`（本地USB/内置摄像头） |
| `enabled` | bool | `true` | 是否启用该摄像头 |

**type=rtsp 时额外字段：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rtsp_url` | string | **必填** | RTSP 流地址 |

**type=local 时额外字段：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device_id` | int | `0` | OpenCV 摄像头设备 ID，`0`=默认摄像头 |

**示例：**

```yaml
cameras:
  # RTSP 网络摄像头
  - id: "gate"
    name: "工厂大门"
    type: "rtsp"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    enabled: true

  # 本地 USB 摄像头
  - id: "local_test"
    name: "本地测试"
    type: "local"
    device_id: 0
    enabled: true
```

### alert — 告警参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cooldown_seconds` | float | `30` | 同一摄像头两次告警的最小间隔（秒）。防止同一根烟产生几十条告警 |
| `min_detection_count` | int | `3` | 连续检测到抽烟的帧数阈值。只有连续 N 帧都检测到抽烟才触发告警，防止单帧误报 |
| `save_dir` | string | `"alerts"` | 关键帧保存根目录，相对于项目根目录。帧保存为 `{save_dir}/{camera_id}/YYYYMMDD_HHMMSS.jpg` |

#### alert.webhook — Webhook 推送

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | string | `null` | Webhook 接收地址。为 null 时不推送，只保存帧 |
| `timeout` | float | `10` | 单次请求超时（秒） |
| `retries` | int | `2` | 失败后重试次数（不含首次） |

### log — 日志参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `file` | string/null | `null` | 日志文件路径。null 表示只输出到控制台 |

## 告警流程

```
读取帧 → 模型推理 → 检测到抽烟？
                         ↓ 是
                   连续帧计数器 +1
                         ↓
                   达到 min_detection_count？
                         ↓ 是
                   冷却期已过？
                         ↓ 是
                   🚨 触发告警
                   ├── 标注帧 + 时间戳 → 保存 JPG
                   └── 构建 payload → Webhook POST
```

**关键设计：**
- **连续帧确认**：只有连续 `min_detection_count` 帧（默认 3 帧）都检测到抽烟才触发，杜绝单帧噪点误报
- **冷却期**：同一摄像头 `cooldown_seconds` 秒内（默认 30 秒）只告警一次，避免告警风暴
- **计数器递减**：没有检测到抽烟时，连续计数器逐步递减（而非直接清零），容忍偶尔丢帧

## Webhook Payload 格式

```json
{
  "camera_id": "gate",
  "camera_name": "工厂大门",
  "timestamp": "2026-06-26T17:30:00",
  "detections": [
    {
      "class": "smoking",
      "confidence": 0.89,
      "bbox": [320, 240, 400, 380]
    }
  ],
  "frame_path": "alerts/gate/20260626_173000.jpg"
}
```

## 测试工具

### Webhook 接收器

`local/webhook_receiver.py` 是一个零依赖的 HTTP 测试服务器，用于接收和验证告警推送：

```bash
# 终端1：启动接收器
python local/webhook_receiver.py

# 终端2：启动检测服务（config.yaml 中 webhook.url 设为 http://localhost:9999/api/smoke-alert）
python -m server.main
```

参数：
- `--port` / `-p`：监听端口（默认 9999）
- `--save` / `-s`：保存收到的告警到 JSON 文件

### 本地摄像头测试

修改 `config.yaml`，启用本地摄像头配置段即可在不连接 RTSP 流的情况下测试。

## 日志级别说明

| 级别 | 内容 | 频率 |
|------|------|------|
| INFO | 启动/停止、告警触发、关键帧保存、每 60 秒运行摘要 | 低频 |
| DEBUG | 每 100 帧的详细统计（FPS、推理延迟）、抽烟检测命中 | 高频 |
| WARNING | RTSP 断线重连、帧读取失败 | 按需 |
| ERROR | 检测异常、Webhook 推送失败、Worker 意外退出 | 按需 |

生产环境推荐使用 `INFO` 级别，调试时使用 `DEBUG`。

## 稳定性设计

### 多层异常防护

每个线程都有顶层 try/except 安全网，确保单个异常不会杀死线程：

```
streamer._update_loop()     ← 顶层 try/except，异常后 1s 恢复
camera_worker._run()        ← 顶层 try/except，异常后跳过当前帧继续
alert_manager.handle()      ← 顶层 try/except，异常后返回 False
webhook.send()              ← 内部重试 + 异常捕获
```

### RTSP 断线重连

RTSP 流断开后自动重连，使用指数退避策略（初始 2 秒，每次失败翻倍，最大 60 秒），连接成功后重置延迟。

### Worker 健康监控

主线程每 30 秒检查所有 Worker 线程是否存活，发现死亡立即记录 ERROR 日志。

### 优雅退出

Ctrl+C → 主线程捕获 KeyboardInterrupt → 依次 stop 所有 Worker → 释放资源 → 退出。不使用 signal 模块（Windows 下与 GPU 线程交互存在已知可靠性问题）。
