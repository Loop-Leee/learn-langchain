"""FastAPI 应用主模块 - 遵循 LangChain 最佳实践"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.runnables import Runnable

from app.config import load_environment
from app.llm import build_checker_chain
from app.regulations import get_regulation, list_check_points, load_regulations
from app.schemas import ContractCheckResponse

# 应用级链实例 - 遵循最佳实践：链应该复用而不是每次请求都创建
_checker_chain: Runnable | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。
    在应用启动时加载配置并初始化链，在关闭时清理资源。
    """
    # 启动时：加载环境变量并初始化链
    global _checker_chain
    load_environment()
    # 预加载法规配置到内存
    load_regulations()
    _checker_chain = build_checker_chain()
    yield
    # 关闭时：清理资源（如果需要）


app = FastAPI(
    title="Contract Check API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/v1/check-points", response_model=List[str])
async def get_check_points():
    """
    获取所有可用的审查要点列表。
    
    Returns:
        审查要点名称列表
    """
    return list_check_points()


@app.post(
    "/v1/contract-check",
    response_model=ContractCheckResponse,
    response_model_by_alias=True,  # 输出中文字段：审查结果/审查过程/审查建议
)
async def contract_check(
    check_point: str = Form(..., description="审查要点（如：主体资格审查）"),
    contract: str = Form(..., description="合同内容文本"),
):
    check_point_text = check_point.strip()
    contract_text = contract.strip()

    if not check_point_text:
        raise HTTPException(status_code=400, detail="check_point 不能为空")
    if not contract_text:
        raise HTTPException(status_code=400, detail="contract 不能为空")

    # 根据审查要点获取对应的法规要求
    try:
        regulation_text = get_regulation(check_point_text)
    except KeyError as e:
        available_points = list_check_points()
        raise HTTPException(
            status_code=400,
            detail=f"未知的审查要点: '{check_point_text}'。可用的审查要点: {available_points}",
        ) from e

    # 使用应用级链实例（在启动时已初始化）
    if _checker_chain is None:
        raise HTTPException(
            status_code=500, detail="Chain not initialized. Please check application startup."
        )

    try:
        # 使用 LangChain 标准接口调用链
        # chain 会自动处理 prompt -> llm -> parser 的流程
        result = _checker_chain.invoke(
            {
                "regulation": regulation_text,
                "contract": contract_text,
            }
        )
        # PydanticOutputParser 已经返回 ContractCheckResponse 对象
        return result
    except ValueError as e:
        # PydanticOutputParser 解析失败
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM response: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM contract-check failed: {type(e).__name__}: {e}",
        ) from e


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import os
    import sys

    # 在导入 uvicorn 之前禁用 uvloop
    os.environ["UVLOOP"] = "false"

    # 检查是否在调试模式下运行
    if "pydevd" in sys.modules:
        print("Running in debug mode, forcing asyncio loop")

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
        loop="asyncio"  # 显式指定使用 asyncio 事件循环
    )
