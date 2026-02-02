"""
配置管理模块
负责从环境变量加载配置并验证
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings:
    """应用配置类"""
    
    # OpenAI 配置
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Agent 配置
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # 安全配置
    ALLOWED_DIRECTORIES: list[str] = os.getenv(
        "ALLOWED_DIRECTORIES", "./"
    ).split(",")
    ENABLE_DELETE: bool = os.getenv("ENABLE_DELETE", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> None:
        """验证配置是否完整"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未设置，请在 .env 文件中配置")


# 全局配置实例
settings = Settings()
