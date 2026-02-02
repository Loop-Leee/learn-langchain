"""
日志工具模块
提供统一的日志记录功能
"""

import logging
from typing import Optional


def setup_logger(
    name: str = "file_agent",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    设置并返回日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        
    Returns:
        配置好的 Logger 实例
    """
    # TODO: 实现日志配置
    # 1. 创建 logger
    # 2. 配置格式
    # 3. 添加控制台处理器
    # 4. 如果指定了日志文件，添加文件处理器
    # 5. 返回 logger
    pass
