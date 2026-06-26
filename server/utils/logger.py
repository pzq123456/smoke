"""
结构化日志模块
- 同时输出到控制台和文件
- 统一格式: [时间] [级别] [模块] 消息
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "smoke_detector",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """
    初始化并返回 logger 实例。

    Args:
        name: logger 名称
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径，None 表示只输出到控制台

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 格式
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "smoke_detector") -> logging.Logger:
    """获取已存在的 logger，不存在则返回 root logger。"""
    return logging.getLogger(name)
