# 文件操作 Agent 开发示例

## 项目简介

这是一个基于 LangChain 开发的智能 Agent 示例，能够理解自然语言指令并自动执行文件操作任务。Agent 通过理解用户的自然语言指令，自动选择合适的工具（文件操作工具、Shell 命令工具）来完成文件管理任务。

### 功能示例

**输入：** 帮我删除当前目录下的所有文件名含有 expire 的文件。

**过程：**
1. Agent 理解用户指令，识别任务类型（删除文件）和条件（文件名包含 "expire"）
2. 调用 `list_files` 工具列出当前目录下的所有文件
3. 调用 `search_files_by_name` 工具筛选出文件名包含 "expire" 的文件
4. 调用 `delete_files` 工具（或 `execute_shell_command` 执行 rm 命令）删除目标文件
5. 返回操作结果和删除的文件列表

**输出：** 已成功删除所有含有 expire 的文件，被删除的文件有：file1_expire.txt, expire_data.json, ...

## 项目架构

### 目录结构

```
agent-demo/
├── README.md                 # 项目说明文档（本文档）
├── requirements.txt          # Python 依赖包列表
├── .env.example             # 环境变量配置示例文件
├── .env                     # 环境变量配置文件（需自行创建，不提交到版本控制）
├── .gitignore               # Git 忽略文件配置
├── main.py                  # 主程序入口
│
├── config/                  # 配置模块
│   ├── __init__.py
│   └── settings.py          # 配置管理（读取环境变量、配置验证）
│
├── tools/                   # 工具模块（Agent 可调用的工具）
│   ├── __init__.py
│   ├── file_tools.py        # 文件操作工具集合
│   │                         # - list_files: 列出文件
│   │                         # - search_files_by_name: 搜索文件
│   │                         # - get_file_info: 获取文件信息
│   └── shell_tools.py       # Shell 命令执行工具
│                             # - execute_shell_command: 执行安全命令
│                             # - delete_files: 删除文件
│
├── agent/                   # Agent 模块
│   ├── __init__.py
│   ├── agent_builder.py     # Agent 构建器
│   │                         # - build_file_agent(): 构建并返回 Agent
│   └── prompts.py           # Agent 提示词模板
│                             # - SYSTEM_PROMPT: 系统提示词
│
└── utils/                   # 工具函数模块
    ├── __init__.py
    └── logger.py            # 日志工具
                              # - setup_logger(): 配置日志记录器
```

### 模块职责说明

#### 1. `config/` - 配置管理
- **职责**: 统一管理项目配置，从环境变量读取敏感信息
- **关键文件**: `settings.py` - 使用 Pydantic 进行配置管理和验证

#### 2. `tools/` - 工具集合
- **职责**: 实现 Agent 可以调用的具体操作工具
- **关键文件**:
  - `file_tools.py`: 文件操作相关工具（列出、搜索、获取信息）
  - `shell_tools.py`: Shell 命令执行工具（带安全限制）

#### 3. `agent/` - Agent 核心
- **职责**: 构建和配置 Agent
- **关键文件**:
  - `prompts.py`: 定义 Agent 的系统提示词和行为规范
  - `agent_builder.py`: 组装 LLM、工具和提示词，构建完整的 Agent

#### 4. `utils/` - 工具函数
- **职责**: 提供通用的工具函数
- **关键文件**: `logger.py` - 日志配置和管理

#### 5. `main.py` - 程序入口
- **职责**: 初始化系统，启动交互循环，处理用户输入和输出

## 实现步骤详解

### 第一阶段：环境准备和依赖安装

#### 1.1 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

#### 1.2 安装依赖包
```bash
pip install -r requirements.txt
```

**核心依赖说明：**
- `langchain>=0.1.0`: Agent 框架核心，提供 Agent 构建和执行能力
- `langchain-openai>=0.0.5`: OpenAI 模型集成（或使用 `langchain-anthropic` 等其他 LLM 提供者）
- `python-dotenv>=1.0.0`: 环境变量管理，用于加载 `.env` 文件
- `pydantic>=2.0.0`: 数据验证和设置管理

#### 1.3 配置环境变量
创建 `.env` 文件（参考 `.env.example`）：
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini  # 或 gpt-3.5-turbo
LOG_LEVEL=INFO
```

### 第二阶段：工具开发（Tools）

工具是 Agent 能够执行的具体操作。每个工具需要使用 LangChain 的 `@tool` 装饰器进行注册，以便 Agent 能够识别和调用。

#### 2.1 文件操作工具 (`tools/file_tools.py`)

**实现要点：**
- 使用 `from langchain.tools import tool` 装饰器
- 每个工具函数需要清晰的文档字符串（Agent 会读取这些描述来决定使用哪个工具）
- 返回结构化的数据，便于 Agent 理解和后续处理

**需要实现的工具函数：**

1. **list_files**
   ```python
   @tool
   def list_files(directory_path: str = ".") -> str:
       """
       列出指定目录下的所有文件和子目录。
       
       Args:
           directory_path: 要列出的目录路径，默认为当前目录
       
       Returns:
           格式化的字符串，包含文件列表信息（文件名、路径、大小等）
       """
   ```
   - 使用 `os.listdir()` 或 `pathlib.Path` 列出文件
   - 返回格式化的字符串，包含文件名、路径、大小等信息
   - 处理目录不存在等异常情况

2. **search_files_by_name**
   ```python
   @tool
   def search_files_by_name(pattern: str, directory_path: str = ".") -> str:
       """
       根据文件名模式搜索文件。支持部分匹配（文件名包含指定关键词）。
       
       Args:
           pattern: 要搜索的文件名关键词或模式
           directory_path: 搜索的目录路径，默认为当前目录
       
       Returns:
           匹配的文件列表（包含完整路径）
       """
   ```
   - 递归搜索目录（可选：支持递归搜索子目录）
   - 使用字符串匹配或正则表达式匹配文件名
   - 返回匹配文件的完整路径列表

3. **get_file_info**
   ```python
   @tool
   def get_file_info(file_path: str) -> str:
       """
       获取指定文件的详细信息。
       
       Args:
           file_path: 文件路径
       
       Returns:
           文件的详细信息（大小、修改时间、类型、权限等）
       """
   ```
   - 使用 `os.path` 或 `pathlib` 获取文件信息
   - 返回文件大小、修改时间、类型等信息

#### 2.2 Shell 命令工具 (`tools/shell_tools.py`)

**实现要点：**
- **安全性是首要考虑**：限制可执行的命令类型，防止危险操作
- 使用白名单机制，只允许执行安全的命令
- 验证命令参数，防止路径遍历攻击

**需要实现的工具函数：**

1. **execute_shell_command**
   ```python
   @tool
   def execute_shell_command(command: str) -> str:
       """
       安全执行 Shell 命令。仅支持白名单中的安全命令。
       
       Args:
           command: 要执行的命令字符串
       
       Returns:
           命令执行结果（stdout 或 stderr）
       
       Raises:
           ValueError: 如果命令不在白名单中
       """
   ```
   - **安全机制：**
     - 命令白名单：只允许 `ls`, `find`, `grep`, `cat`, `head`, `tail` 等只读命令
     - 禁止执行：`rm`, `rmdir`, `mv`, `cp` 等修改性命令（使用专门的工具函数）
     - 参数验证：检查命令参数，防止路径遍历（如 `../../../etc/passwd`）
   - 使用 `subprocess.run()` 执行命令
   - 返回 stdout 和 stderr 的内容

2. **delete_files**
   ```python
   @tool
   def delete_files(file_paths: list[str]) -> str:
       """
       删除指定的文件列表。这是一个危险操作，需要谨慎使用。
       
       Args:
           file_paths: 要删除的文件路径列表
       
       Returns:
           删除结果，包含成功和失败的文件列表
       """
   ```
   - 验证文件路径（防止删除系统关键文件）
   - 使用 `os.remove()` 或 `pathlib.Path.unlink()` 删除文件
   - 返回详细的删除结果（成功/失败的文件列表）
   - 处理文件不存在、权限不足等异常情况

### 第三阶段：Agent 构建

#### 3.1 提示词设计 (`agent/prompts.py`)

**系统提示词结构：**

```python
SYSTEM_PROMPT = """你是一个专业的文件操作助手，能够理解用户的自然语言指令并执行文件管理任务。

你的职责：
1. 理解用户的文件操作需求
2. 选择合适的工具来完成任务
3. 按照正确的顺序执行操作
4. 向用户报告操作结果

可用工具：
- list_files: 列出目录下的文件
- search_files_by_name: 根据文件名模式搜索文件
- get_file_info: 获取文件详细信息
- delete_files: 删除指定文件
- execute_shell_command: 执行安全的 Shell 命令（仅限只读命令）

操作流程：
1. 分析用户需求，确定需要执行的操作
2. 如果需要查找文件，先使用 list_files 或 search_files_by_name
3. 确认目标文件后，执行相应的操作（删除、查看等）
4. 汇总操作结果并返回给用户

安全约束：
- 删除操作前，请确认文件列表
- 不要执行危险的系统命令
- 确保操作在用户指定的目录范围内
"""
```

**设计要点：**
- 清晰定义 Agent 的角色和能力
- 列出所有可用工具及其用途
- 提供操作流程指导，帮助 Agent 做出正确决策
- 强调安全约束，防止误操作

#### 3.2 Agent 构建器 (`agent/agent_builder.py`)

**实现步骤：**

1. **初始化 LLM**
   ```python
   from langchain_openai import ChatOpenAI
   
   llm = ChatOpenAI(
       model=settings.OPENAI_MODEL_NAME,
       temperature=0,  # 降低随机性，提高准确性
       api_key=settings.OPENAI_API_KEY
   )
   ```
   - 从配置中读取 API Key 和模型名称
   - 设置合适的 temperature（文件操作需要准确性，建议设为 0）

2. **导入和注册工具**
   ```python
   from tools.file_tools import list_files, search_files_by_name, get_file_info
   from tools.shell_tools import execute_shell_command, delete_files
   
   tools = [
       list_files,
       search_files_by_name,
       get_file_info,
       delete_files,
       execute_shell_command,
   ]
   ```
   - 导入所有工具函数（已使用 `@tool` 装饰器）
   - 将工具添加到列表中

3. **创建提示词模板**
   ```python
   from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
   
   prompt = ChatPromptTemplate.from_messages([
       ("system", SYSTEM_PROMPT),
       ("human", "{input}"),
       MessagesPlaceholder(variable_name="agent_scratchpad"),
   ])
   ```
   - 使用 `ChatPromptTemplate` 创建提示词模板
   - 包含系统提示词、用户输入和 Agent 思考过程占位符

4. **构建 Agent**
   ```python
   from langchain.agents import create_openai_tools_agent
   
   agent = create_openai_tools_agent(llm, tools, prompt)
   ```
   - 使用 `create_openai_tools_agent` 创建 Agent（适用于支持工具调用的模型）
   - 或使用 `create_react_agent`（适用于其他模型）

5. **创建 AgentExecutor**
   ```python
   from langchain.agents import AgentExecutor
   
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True,  # 显示执行过程
       max_iterations=10,  # 最大迭代次数，防止无限循环
       handle_parsing_errors=True,  # 处理解析错误
   )
   ```
   - 配置 `verbose=True` 以便调试和查看执行过程
   - 设置合理的 `max_iterations`（建议 5-10 次）
   - 启用错误处理机制

6. **返回 AgentExecutor**
   ```python
   return agent_executor
   ```

### 第四阶段：主程序开发

#### 4.1 配置管理 (`config/settings.py`)

**实现要点：**
- 使用 `pydantic` 的 `BaseSettings` 进行配置管理
- 从环境变量和 `.env` 文件读取配置
- 提供配置验证和默认值

**需要实现的配置项：**

```python
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

class Settings(BaseSettings):
    # OpenAI 配置
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    
    # 安全配置
    ALLOWED_COMMANDS: list[str] = ["ls", "find", "grep", "cat", "head", "tail"]
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

**配置验证：**
- 验证 `OPENAI_API_KEY` 是否存在
- 验证模型名称是否有效
- 提供友好的错误提示

#### 4.2 日志工具 (`utils/logger.py`)

**实现要点：**
- 配置日志格式和级别
- 支持文件和控制台输出

```python
import logging
from config.settings import settings

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    设置并返回配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

#### 4.3 主程序 (`main.py`)

**实现主程序流程：**

1. **初始化阶段**
   ```python
   def main():
       # 1. 验证配置
       if not settings.OPENAI_API_KEY:
           print("错误: 未设置 OPENAI_API_KEY，请在 .env 文件中配置")
           return
       
       # 2. 初始化日志
       logger = setup_logger()
       logger.info("文件操作 Agent 启动")
       
       # 3. 构建 Agent
       logger.info("正在构建 Agent...")
       agent = build_file_agent()
       logger.info("Agent 构建完成")
   ```

2. **交互循环**
   ```python
       # 4. 进入交互循环
       print("=" * 50)
       print("文件操作 Agent 已就绪")
       print("输入 'quit' 或 'exit' 退出")
       print("=" * 50)
       
       while True:
           try:
               # 接收用户输入
               user_input = input("\n请输入您的指令: ").strip()
               
               if user_input.lower() in ['quit', 'exit', '退出']:
                   print("再见！")
                   break
               
               if not user_input:
                   continue
               
               # 调用 Agent 执行任务
               logger.info(f"用户输入: {user_input}")
               result = agent.invoke({"input": user_input})
               
               # 显示执行结果
               print("\n" + "=" * 50)
               print("执行结果:")
               print("=" * 50)
               print(result.get("output", "无输出"))
               print("=" * 50)
               
           except KeyboardInterrupt:
               print("\n\n程序被用户中断")
               break
           except Exception as e:
               logger.error(f"执行出错: {e}", exc_info=True)
               print(f"\n错误: {e}")
   ```

3. **结果展示优化**
   - 格式化输出，使结果更易读
   - 显示 Agent 的执行步骤（如果 verbose=True）
   - 提供操作摘要（如删除了多少个文件）

### 第五阶段：安全性和错误处理

#### 5.1 命令白名单机制

**实现位置：** `tools/shell_tools.py` 的 `execute_shell_command` 函数

```python
ALLOWED_COMMANDS = ["ls", "find", "grep", "cat", "head", "tail", "wc", "stat"]

def execute_shell_command(command: str) -> str:
    # 解析命令
    parts = command.strip().split()
    cmd = parts[0] if parts else ""
    
    # 检查命令是否在白名单中
    if cmd not in ALLOWED_COMMANDS:
        raise ValueError(f"不允许执行命令: {cmd}。仅支持只读命令。")
    
    # 执行命令...
```

**安全规则：**
- 只允许执行只读命令（`ls`, `find`, `grep` 等）
- 禁止执行修改性命令（`rm`, `mv`, `cp`, `chmod` 等）
- 禁止执行系统管理命令（`sudo`, `su`, `shutdown` 等）

#### 5.2 路径验证机制

**实现位置：** 所有涉及文件路径的工具函数

```python
import os
from pathlib import Path

def validate_path(file_path: str, base_dir: str = ".") -> Path:
    """
    验证文件路径是否安全
    
    Args:
        file_path: 要验证的文件路径
        base_dir: 基础目录，默认为当前目录
    
    Returns:
        规范化的 Path 对象
    
    Raises:
        ValueError: 如果路径不安全
    """
    base = Path(base_dir).resolve()
    target = (base / file_path).resolve()
    
    # 防止路径遍历攻击
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(f"路径不安全: {file_path}")
    
    return target
```

**验证规则：**
- 防止路径遍历（如 `../../../etc/passwd`）
- 确保操作在指定目录范围内
- 规范化路径，处理相对路径和绝对路径

#### 5.3 异常处理

**在各工具函数中实现：**

```python
@tool
def delete_files(file_paths: list[str]) -> str:
    """删除文件"""
    results = {"success": [], "failed": []}
    
    for file_path in file_paths:
        try:
            # 验证路径
            validated_path = validate_path(file_path)
            
            # 检查文件是否存在
            if not validated_path.exists():
                results["failed"].append(f"{file_path} (文件不存在)")
                continue
            
            # 执行删除
            validated_path.unlink()
            results["success"].append(file_path)
            
        except PermissionError:
            results["failed"].append(f"{file_path} (权限不足)")
        except Exception as e:
            results["failed"].append(f"{file_path} (错误: {str(e)})")
    
    # 返回格式化的结果
    return format_delete_results(results)
```

**异常类型处理：**
- `FileNotFoundError`: 文件不存在
- `PermissionError`: 权限不足
- `ValueError`: 路径不安全或参数无效
- `OSError`: 其他系统错误

#### 5.4 日志记录

**实现要点：**
- 记录所有工具调用（包括参数和结果）
- 记录 Agent 的决策过程
- 记录错误和异常信息
- 支持不同日志级别（DEBUG, INFO, WARNING, ERROR）

**在工具函数中添加日志：**

```python
from utils.logger import setup_logger

logger = setup_logger(__name__)

@tool
def delete_files(file_paths: list[str]) -> str:
    logger.info(f"删除文件请求: {file_paths}")
    # ... 执行删除操作
    logger.info(f"删除完成: 成功 {len(results['success'])} 个，失败 {len(results['failed'])} 个")
    return result
```

### 第六阶段：测试和优化

#### 6.1 单元测试

**测试工具函数：**

创建 `tests/` 目录，编写单元测试：

```python
# tests/test_file_tools.py
import pytest
from tools.file_tools import list_files, search_files_by_name

def test_list_files():
    """测试列出文件功能"""
    result = list_files(".")
    assert isinstance(result, str)
    assert len(result) > 0

def test_search_files_by_name():
    """测试文件搜索功能"""
    # 创建测试文件
    test_file = Path("test_expire_file.txt")
    test_file.write_text("test")
    
    try:
        result = search_files_by_name("expire", ".")
        assert "test_expire_file.txt" in result
    finally:
        test_file.unlink()
```

**测试要点：**
- 测试正常情况
- 测试边界情况（空目录、不存在的路径等）
- 测试异常处理

#### 6.2 集成测试

**测试完整场景：**

```python
# tests/test_agent_integration.py
def test_delete_files_scenario():
    """测试删除文件的完整场景"""
    # 1. 创建测试文件
    test_files = [
        "file1_expire.txt",
        "file2_expire.json",
        "normal_file.txt"
    ]
    for f in test_files:
        Path(f).write_text("test")
    
    try:
        # 2. 构建 Agent
        agent = build_file_agent()
        
        # 3. 执行任务
        result = agent.invoke({
            "input": "删除当前目录下所有文件名包含 expire 的文件"
        })
        
        # 4. 验证结果
        assert "成功" in result["output"] or "删除" in result["output"]
        assert not Path("file1_expire.txt").exists()
        assert not Path("file2_expire.json").exists()
        assert Path("normal_file.txt").exists()  # 不应被删除
        
    finally:
        # 清理
        for f in test_files:
            if Path(f).exists():
                Path(f).unlink()
```

#### 6.3 性能优化

**优化策略：**

1. **减少 API 调用**
   - 优化提示词，让 Agent 更准确地选择工具
   - 设置合理的 `max_iterations`，避免无限循环
   - 缓存文件列表结果（如果目录未变化）

2. **优化工具执行**
   - 批量操作而非逐个处理
   - 使用更高效的文件操作方式

3. **错误处理优化**
   - 快速失败：尽早发现并报告错误
   - 提供清晰的错误信息，减少重试次数

## 技术栈

- **LangChain**: Agent 框架
- **OpenAI API**: LLM 模型（或使用其他提供者如 Anthropic、本地模型等）
- **Python 3.8+**: 开发语言

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件（参考 `.env.example`）：

```bash
cp .env.example .env
# 然后编辑 .env 文件，填入你的 OPENAI_API_KEY
```

### 3. 运行程序

```bash
python main.py
```

### 4. 使用示例

在交互界面中输入自然语言指令：

```
请输入您的指令: 帮我删除当前目录下的所有文件名含有 expire 的文件
```

Agent 会自动：
1. 理解你的指令
2. 列出当前目录的文件
3. 筛选出包含 "expire" 的文件
4. 删除这些文件
5. 返回操作结果

## 使用示例（代码方式）

```python
from agent.agent_builder import build_file_agent

# 初始化 Agent
agent = build_file_agent()

# 执行任务
result = agent.invoke({
    "input": "帮我删除当前目录下的所有文件名含有 expire 的文件"
})

print(result["output"])
```

## 实施步骤总结

### 阶段一：基础搭建（已完成 ✅）
- [x] 项目架构设计
- [x] 目录结构创建
- [x] 基础文件框架
- [x] README 文档编写

### 阶段二：核心功能开发（待实现 ⏳）

**实施顺序建议：**

```
1. config/settings.py (配置管理)
   ↓
2. utils/logger.py (日志工具)
   ↓
3. tools/file_tools.py (文件操作工具)
   ↓
4. tools/shell_tools.py (Shell 命令工具)
   ↓
5. agent/prompts.py (提示词设计)
   ↓
6. agent/agent_builder.py (Agent 构建)
   ↓
7. main.py (主程序)
```

**说明：** 按照依赖关系顺序实现，每个模块完成后可以单独测试。

#### 步骤 1: 配置管理 (`config/settings.py`)
- [ ] 实现 `Settings` 类，使用 Pydantic BaseSettings
- [ ] 从环境变量读取配置
- [ ] 添加配置验证逻辑

#### 步骤 2: 日志工具 (`utils/logger.py`)
- [ ] 实现 `setup_logger` 函数
- [ ] 配置日志格式和处理器

#### 步骤 3: 文件操作工具 (`tools/file_tools.py`)
- [ ] 实现 `list_files` 工具（使用 `@tool` 装饰器）
- [ ] 实现 `search_files_by_name` 工具
- [ ] 实现 `get_file_info` 工具
- [ ] 添加路径验证和错误处理

#### 步骤 4: Shell 命令工具 (`tools/shell_tools.py`)
- [ ] 实现 `execute_shell_command` 工具
- [ ] 实现命令白名单机制
- [ ] 实现 `delete_files` 工具
- [ ] 添加安全验证（路径验证、命令限制）

#### 步骤 5: Agent 提示词 (`agent/prompts.py`)
- [ ] 设计并实现 `SYSTEM_PROMPT`
- [ ] 包含工具说明、操作流程、安全约束

#### 步骤 6: Agent 构建器 (`agent/agent_builder.py`)
- [ ] 实现 `build_file_agent` 函数
- [ ] 初始化 LLM（ChatOpenAI）
- [ ] 导入并注册所有工具
- [ ] 创建提示词模板
- [ ] 构建 Agent 和 AgentExecutor

#### 步骤 7: 主程序 (`main.py`)
- [ ] 实现配置验证
- [ ] 实现日志初始化
- [ ] 实现 Agent 构建
- [ ] 实现交互循环
- [ ] 实现结果展示和错误处理

### 阶段三：完善和优化（待实现）
- [ ] 添加单元测试
- [ ] 添加集成测试
- [ ] 性能优化
- [ ] 文档完善

## 开发计划检查清单

- [x] 项目架构设计
- [x] README 文档编写
- [ ] 环境配置和依赖安装
- [ ] 工具开发（文件操作、Shell 命令）
- [ ] Agent 构建和提示词设计
- [ ] 主程序开发
- [ ] 安全性和错误处理
- [ ] 测试和优化

## 注意事项

1. **安全性**: 文件删除操作不可逆，需要谨慎处理
2. **权限**: 确保 Agent 有足够的权限执行文件操作
3. **API 成本**: 注意 LLM API 调用的成本控制
4. **错误处理**: 完善的错误处理机制确保系统稳定性

## 后续扩展方向

1. 支持更多文件操作（复制、移动、重命名等）
2. 支持批量操作和模式匹配
3. 添加操作确认机制
4. 支持更复杂的查询（如按文件大小、修改时间筛选）
5. 添加操作历史记录功能
