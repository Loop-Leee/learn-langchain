from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.schemas import ContractCheckResponse


def _load_env() -> None:
    # 兼容：同目录常用命名 example.env；也兼容用户自己创建 .env（若未被忽略）
    load_dotenv("example.env", override=False)
    load_dotenv(".env", override=False)


def get_deepseek_llm(temperature: float = 0.0) -> ChatOpenAI:
    _load_env()

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY (or OPENAI_API_KEY). "
            "Please set it in environment variables or example.env."
        )

    # DeepSeek 提供 OpenAI 兼容接口；langchain-openai 可直接用 base_url + api_key
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def build_checker_chain(llm: Optional[ChatOpenAI] = None):
    if llm is None:
        llm = get_deepseek_llm()

    # 让模型直接按 Pydantic Schema 输出结构化结果
    return llm.with_structured_output(ContractCheckResponse)


