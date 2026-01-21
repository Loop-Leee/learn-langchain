"""LLM 和链的构建模块 - 遵循 LangChain 最佳实践"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import LLMConfig, ModelType
from app.schemas import ContractCheckResponse


def create_chat_model(
    config: Optional[LLMConfig] = None, 
    model_type: str | ModelType = ModelType.AUTO
) -> BaseChatModel:
    """
    创建 ChatModel 实例。
    
    遵循 LangChain 最佳实践：
    - 使用抽象类型 BaseChatModel 而不是具体类型
    - 通过配置对象传递参数
    - 支持依赖注入以便测试
    - 使用策略模式选择模型配置
    
    Args:
        config: LLM 配置对象，如果为 None 则根据 model_type 从环境变量创建
        model_type: 模型类型，可以是字符串或 ModelType 枚举，可选值：
            - "auto" 或 ModelType.AUTO: 自动选择（默认使用 DeepSeek）
            - "deepseek" 或 ModelType.DEEPSEEK: 使用 DeepSeek 配置
            - "qwen3" 或 ModelType.QWEN3: 使用 Qwen3 配置
        
    Returns:
        BaseChatModel 实例（具体为 ChatOpenAI）
        
    Examples:
        # 使用默认配置（DeepSeek）
        llm = create_chat_model()
        
        # 切换到 Qwen3（使用字符串）
        llm = create_chat_model(model_type="qwen3")
        
        # 切换到 Qwen3（使用枚举）
        llm = create_chat_model(model_type=ModelType.QWEN3)
        
        # 使用自定义配置
        config = LLMConfig.from_qwen3_env()
        llm = create_chat_model(config=config)
    """
    if config is None:
        # 使用策略模式创建配置
        model_type = os.getenv("MODEL_TYPE", ModelType.DEEPSEEK.value) or ModelType.AUTO
        config = LLMConfig.from_model_type(model_type)

    # DeepSeek 和 Qwen3 都提供 OpenAI 兼容接口；langchain-openai 可直接用 base_url + api_key
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )


def build_checker_chain(llm: Optional[BaseChatModel] = None) -> Runnable:
    """
使用 LangChain 标准接口构建合同审查链。
    
    遵循 LangChain 最佳实践：
    - 使用 LCEL (LangChain Expression Language) 构建链
    - 使用 PydanticOutputParser 确保跨模型的一致性
    - 返回 Runnable 抽象类型，支持链的复用和组合
    - 支持依赖注入以便测试
    
    Args:
        llm: BaseChatModel 实例，如果为 None 则从配置创建
        
    Returns:
        Runnable 链，输入为 {"regulation": str, "contract": str}，
        输出为 ContractCheckResponse
    """
    if llm is None:
        llm = create_chat_model()

    # 使用 LangChain 的 PydanticOutputParser 来解析结构化输出
    output_parser = PydanticOutputParser(pydantic_object=ContractCheckResponse)

    # 构建系统提示，包含输出格式说明
    system_prompt = """你是一名严谨的合同合规审查助手。
你的任务：根据"法规要求"逐条核对"合同内容"是否满足。

判定规则：
1) 对于不合格：必须指出缺失/不明确之处（对应到具体条款点），并给出可执行的补充建议。
2) 只输出结构化结果，不要输出多余字段，不要输出 Markdown。
3) 不要编造合同中不存在的信息。
4) 特别注意[最高优先级]/[特殊情况]等合规条件的特殊判断。

{format_instructions}"""

    # 构建用户提示模板
    user_prompt_template = """请按法规要求审查合同内容，并输出结构化结果。

法规要求：
{regulation}

合同内容：
{contract}"""

    # 使用 LangChain 的 ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt_template),
        ]
    ).partial(format_instructions=output_parser.get_format_instructions())

    # 使用 LangChain Expression Language (LCEL) 构建链
    # 链的流程：prompt -> llm -> output_parser
    chain = prompt | llm | output_parser

    return chain


