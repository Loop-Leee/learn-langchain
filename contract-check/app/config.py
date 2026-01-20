"""配置管理模块 - 遵循 LangChain 最佳实践：在应用启动时加载配置"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def load_environment() -> None:
    """
    加载环境变量。
    应该在应用启动时调用一次，而不是每次使用配置时调用。
    """
    # 兼容：同目录常用命名 example.env；也兼容用户自己创建 .env（若未被忽略）
    load_dotenv("example.env", override=False)
    load_dotenv(".env", override=False)


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置类 - 使用不可变数据类确保配置一致性"""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0

    @classmethod
    def from_env(cls, temperature: float = 0.0) -> LLMConfig:
        """
        从环境变量创建配置。
        
        Args:
            temperature: 模型温度参数
            
        Returns:
            LLMConfig 实例
            
        Raises:
            RuntimeError: 如果缺少必需的 API key
        """
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not api_key:
            raise RuntimeError(
                "Missing DEEPSEEK_API_KEY (or OPENAI_API_KEY). "
                "Please set it in environment variables or example.env."
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )
