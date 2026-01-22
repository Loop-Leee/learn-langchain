"""LLM 和链的构建模块 - 遵循 LangChain 最佳实践"""

from __future__ import annotations

import os
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import Field

from app.config import LLMConfig, ModelType
from app.schemas import ContractCheckResponse


class ChatFarui(BaseChatModel):
    """
    通义法睿 Chat Model - 封装 DashScope 原生 SDK。
    
    通义法睿是阿里云的法律专用模型，不支持 OpenAI 兼容接口，
    必须通过 DashScope SDK 的 Generation.call() 方法调用。
    
    Examples:
        llm = ChatFarui(api_key="your-api-key", model="farui-plus")
        response = llm.invoke([HumanMessage(content="请帮我生成一份起诉书")])
    """
    
    api_key: str = Field(..., description="DashScope API Key")
    model: str = Field(default="farui-plus", description="模型名称")
    temperature: float = Field(default=0.5, ge=0.0, le=2.0, description="采样温度")
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        return "farui"
    
    @property
    def _identifying_params(self) -> dict[str, Any]:
        """返回用于标识此 LLM 的参数"""
        return {
            "model": self.model,
            "temperature": self.temperature,
        }
    
    def _convert_messages_to_dashscope_format(
        self, messages: List[BaseMessage]
    ) -> List[dict[str, str]]:
        """
        将 LangChain 消息格式转换为 DashScope 格式。
        
        Args:
            messages: LangChain 消息列表
            
        Returns:
            DashScope 格式的消息列表
        """
        dashscope_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                dashscope_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                dashscope_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                dashscope_messages.append({"role": "assistant", "content": msg.content})
            else:
                # 其他类型的消息作为 user 消息处理
                dashscope_messages.append({"role": "user", "content": str(msg.content)})
        return dashscope_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        调用 DashScope API 生成回复。
        
        Args:
            messages: 输入消息列表
            stop: 停止词列表（可选）
            run_manager: 回调管理器（可选）
            **kwargs: 其他参数
            
        Returns:
            ChatResult 包含生成的回复
        """
        try:
            import dashscope
            from dashscope import Generation
        except ImportError as e:
            raise ImportError(
                "dashscope 包未安装。请运行: pip install dashscope"
            ) from e
        
        # 设置 API Key
        dashscope.api_key = self.api_key
        
        # 转换消息格式
        dashscope_messages = self._convert_messages_to_dashscope_format(messages)
        
        # 调用 DashScope API
        response = Generation.call(
            model=self.model,
            messages=dashscope_messages,
            result_format="message",
            temperature=self.temperature,
            **kwargs,
        )
        
        # 检查响应状态
        if response.status_code != 200:
            raise ValueError(
                f"DashScope API 调用失败: {response.code} - {response.message}"
            )
        
        # 提取回复内容
        content = response.output.choices[0]["message"]["content"]
        
        # 构造 ChatResult
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        流式调用 DashScope API。
        
        Args:
            messages: 输入消息列表
            stop: 停止词列表（可选）
            run_manager: 回调管理器（可选）
            **kwargs: 其他参数
            
        Yields:
            ChatGenerationChunk 流式输出块
        """
        try:
            import dashscope
            from dashscope import Generation
        except ImportError as e:
            raise ImportError(
                "dashscope 包未安装。请运行: pip install dashscope"
            ) from e
        
        # 设置 API Key
        dashscope.api_key = self.api_key
        
        # 转换消息格式
        dashscope_messages = self._convert_messages_to_dashscope_format(messages)
        
        # 流式调用 DashScope API
        responses = Generation.call(
            model=self.model,
            messages=dashscope_messages,
            result_format="message",
            temperature=self.temperature,
            stream=True,
            incremental_output=True,  # 增量输出
            **kwargs,
        )
        
        for response in responses:
            if response.status_code != 200:
                raise ValueError(
                    f"DashScope API 调用失败: {response.code} - {response.message}"
                )
            
            content = response.output.choices[0]["message"]["content"]
            chunk = ChatGenerationChunk(message=AIMessage(content=content))
            
            if run_manager:
                run_manager.on_llm_new_token(content)
            
            yield chunk


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
            - "farui" 或 ModelType.FARUI: 使用通义法睿配置（法律专用模型）
        
    Returns:
        BaseChatModel 实例（ChatOpenAI 或 ChatFarui）
        
    Examples:
        # 使用默认配置（DeepSeek）
        llm = create_chat_model()
        
        # 切换到 Qwen3（使用字符串）
        llm = create_chat_model(model_type="qwen3")
        
        # 切换到 Qwen3（使用枚举）
        llm = create_chat_model(model_type=ModelType.QWEN3)
        
        # 使用通义法睿（法律专用模型）
        llm = create_chat_model(model_type="farui")
        
        # 使用自定义配置
        config = LLMConfig.from_qwen3_env()
        llm = create_chat_model(config=config)
    """
    if config is None:
        # 使用策略模式创建配置
        model_type = os.getenv("MODEL_TYPE", ModelType.DEEPSEEK.value) or ModelType.AUTO
        config = LLMConfig.from_model_type(model_type)

    # 检查是否需要使用原生 DashScope SDK（法睿模型）
    if config.use_native_dashscope:
        return ChatFarui(
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
        )
    
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
        print(f"已创建模型：{llm.model_name}, 温度：{llm.temperature}")

    # 使用 LangChain 的 PydanticOutputParser 来解析结构化输出
    output_parser = PydanticOutputParser(pydantic_object=ContractCheckResponse)

    # 构建系统提示，包含输出格式说明
    system_prompt = """你是一个法律合同审查助手。
你的任务：逐条核对"合同内容"是否满足"法规要求", 最后再给出[合格/不合格]结论。
你的回答需要基于以下规则：
1. 必须严格依据用户提供的规则进行判断。
2. 禁止自行补充合同未出现的条件。
3. 回答完成后回顾自己的回答是否存在逻辑错误并指正。

{format_instructions}"""

    # 构建用户提示模板
    user_prompt_template = """

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


