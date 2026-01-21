"""配置管理模块 - 遵循 LangChain 最佳实践：在应用启动时加载配置"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

# 项目根目录（contract-check 目录）
_PROJECT_ROOT = Path(__file__).parent.parent


def load_environment() -> None:
    """
    加载环境变量。
    应该在应用启动时调用一次，而不是每次使用配置时调用。
    
    加载顺序：
    1. 先加载 example.env 作为默认配置（不覆盖已有环境变量）
    2. 再加载 .env，覆盖 example.env 中的配置（用户自定义配置优先）
    """
    # 使用项目根目录的绝对路径加载配置文件
    example_env = _PROJECT_ROOT / "example.env"
    dotenv_file = _PROJECT_ROOT / ".env"
    
    # 先加载 example.env 作为默认值
    load_dotenv(example_env, override=False)
    # 再加载 .env，允许覆盖 example.env 中的配置
    load_dotenv(dotenv_file, override=True)


class ModelType(str, Enum):
    """模型类型枚举 - 策略模式"""
    DEEPSEEK = "deepseek"
    QWEN3 = "qwen3"
    FARUI = "farui"  # 通义法睿 - 法律专用模型
    AUTO = "auto"  # 默认使用 DeepSeek


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置类 - 使用不可变数据类确保配置一致性"""

    api_key: str
    base_url: str | None  # 法睿模型不需要 base_url
    model: str
    temperature: float = 0.0
    use_native_dashscope: bool = False  # 是否使用 DashScope 原生 SDK（法睿需要）

    @classmethod
    def from_deepseek_env(cls, temperature: float = 0.5) -> LLMConfig:
        """
        从环境变量创建 DeepSeek 配置（策略：DeepSeek）。
        
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

    @classmethod
    def from_qwen3_env(cls, temperature: float = 0.5) -> LLMConfig:
        """
        从环境变量创建 Qwen3 配置（策略：Qwen3）。
        
        Qwen3 提供 OpenAI 兼容接口，可以使用 ChatOpenAI 直接调用。
        
        Args:
            temperature: 模型温度参数
            
        Returns:
            LLMConfig 实例
            
        Raises:
            RuntimeError: 如果缺少必需的 API key
        """
        api_key = os.getenv("QWEN3_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("QWEN3_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("QWEN3_MODEL", "qwen-plus")

        if not api_key:
            raise RuntimeError(
                "Missing QWEN3_API_KEY (or QWEN_API_KEY or OPENAI_API_KEY). "
                "Please set it in environment variables or example.env."
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    @classmethod
    def from_farui_env(cls, temperature: float = 0.2) -> LLMConfig:
        """
        从环境变量创建通义法睿配置（策略：Farui）。
        
        通义法睿是阿里云的法律专用模型，不支持 OpenAI 兼容接口，
        必须使用 DashScope 原生 SDK 调用。
        
        Args:
            temperature: 模型温度参数
            
        Returns:
            LLMConfig 实例
            
        Raises:
            RuntimeError: 如果缺少必需的 API key
        """
        api_key = os.getenv("FARUI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN3_API_KEY")
        model = os.getenv("FARUI_MODEL", "farui-plus")

        if not api_key:
            raise RuntimeError(
                "Missing FARUI_API_KEY (or DASHSCOPE_API_KEY). "
                "Please set it in environment variables or .env file."
            )

        return cls(
            api_key=api_key,
            base_url=None,  # 法睿不使用 base_url
            model=model,
            temperature=temperature,
            use_native_dashscope=True,  # 标记使用原生 DashScope SDK
        )

    @classmethod
    def from_model_type(cls, model_type: str | ModelType = ModelType.AUTO, temperature: float = 0.2) -> LLMConfig:
        """
        策略模式：根据模型类型从环境变量创建配置。
        
        Args:
            model_type: 模型类型，可以是字符串或 ModelType 枚举
            temperature: 模型温度参数
            
        Returns:
            LLMConfig 实例
            
        Raises:
            ValueError: 如果模型类型不支持
            RuntimeError: 如果缺少必需的 API key
            
        Examples:
            # 使用枚举
            config = LLMConfig.from_model_type(ModelType.QWEN3)
            
            # 使用字符串
            config = LLMConfig.from_model_type("qwen3")
            config = LLMConfig.from_model_type("deepseek")
        """
        # 策略映射表
        _strategy_map: dict[str, Callable[[float], LLMConfig]] = {
            ModelType.DEEPSEEK: cls.from_deepseek_env,
            ModelType.QWEN3: cls.from_qwen3_env,
            ModelType.FARUI: cls.from_farui_env,
            ModelType.AUTO: cls.from_deepseek_env,  # 默认使用 DeepSeek
        }
        
        # 统一转换为字符串（支持枚举和字符串输入）
        model_type_str = model_type.value if isinstance(model_type, ModelType) else str(model_type).lower()
        
        # 获取策略函数
        strategy = _strategy_map.get(model_type_str)
        if strategy is None:
            available_types = ", ".join([t.value for t in ModelType])
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available types: {available_types}"
            )
        
        # 执行策略
        return strategy(temperature)
