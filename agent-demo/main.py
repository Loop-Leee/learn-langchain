"""
主程序入口
提供交互式界面，接收用户输入并执行 Agent 任务
"""

from config.settings import settings
from agent.agent_builder import build_file_agent
from utils.logger import setup_logger


def main():
    """
    主函数
    初始化 Agent 并进入交互循环
    """
    # TODO: 实现主程序逻辑
    # 1. 验证配置
    # 2. 初始化日志
    # 3. 构建 Agent
    # 4. 进入交互循环：
    #    - 接收用户输入
    #    - 调用 Agent 执行任务
    #    - 显示执行结果
    #    - 处理异常
    pass


if __name__ == "__main__":
    main()
