"""LLM 和链的构建模块 - 遵循 LangChain 最佳实践"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import LLMConfig
from app.schemas import ContractCheckResponse


def create_chat_model(config: Optional[LLMConfig] = None) -> BaseChatModel:
    """
    创建 BaseChatModel 实例。
    
    遵循 LangChain 最佳实践：
    - 使用抽象类型 BaseChatModel 而不是具体类型
    - 通过配置对象传递参数
    - 支持依赖注入以便测试
    
    Args:
        config: LLM 配置对象，如果为 None 则从环境变量创建
        
    Returns:
        BaseChatModel 实例（具体为 ChatOpenAI）
    """
    if config is None:
        config = LLMConfig.from_env()

    # DeepSeek 提供 OpenAI 兼容接口；langchain-openai 可直接用 base_url + api_key
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
1) 仅当法规要求中的每一条都被合同内容明确满足时，才输出"合格"；否则输出"不合格"。
2) 对于不合格：必须指出缺失/不明确之处（对应到具体条款点），并给出可执行的补充建议。
3) 只输出结构化结果，不要输出多余字段，不要输出 Markdown。
4) 不要编造合同中不存在的信息；如果合同未提供某字段/信息，视为不满足。

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


