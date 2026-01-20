"""FastAPI 应用主模块 - 遵循 LangChain 最佳实践"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.runnables import Runnable

from app.config import load_environment
from app.llm import build_checker_chain
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
    _checker_chain = build_checker_chain()
    yield
    # 关闭时：清理资源（如果需要）


app = FastAPI(
    title="Contract Check API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post(
    "/v1/contract-check",
    response_model=ContractCheckResponse,
    response_model_by_alias=True,  # 输出中文字段：审查结果/审查过程/审查建议
)
async def contract_check(
    regulation: str = Form(..., description="法规要求文本"),
    contract: str = Form(..., description="合同内容文本"),
):
    regulation_text = regulation.strip()
    contract_text = contract.strip()

    if not regulation_text:
        raise HTTPException(status_code=400, detail="regulation 不能为空")
    if not contract_text:
        raise HTTPException(status_code=400, detail="contract 不能为空")

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
