"""
Agent 构建器
负责初始化 LLM、注册工具、构建 Agent
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.settings import settings
from agent.prompts import SYSTEM_PROMPT


def build_file_agent():
    """
    构建文件操作 Agent
    
    Returns:
        配置好的 AgentExecutor 实例
    """
    # TODO: 实现 Agent 构建逻辑
    # 1. 初始化 LLM (ChatOpenAI)
    # 2. 导入并注册工具（file_tools, shell_tools）
    # 3. 创建提示词模板
    # 4. 构建 Agent
    # 5. 创建 AgentExecutor
    # 6. 返回 AgentExecutor
    pass
