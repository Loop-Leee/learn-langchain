from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import build_checker_chain
from app.schemas import ContractCheckResponse

app = FastAPI(title="Contract Check API", version="0.1.0")


SYSTEM_PROMPT = """你是一名严谨的合同合规审查助手。
你的任务：根据“法规要求”逐条核对“合同内容”是否满足。

判定规则：
1) 仅当法规要求中的每一条都被合同内容明确满足时，才输出“合格”；否则输出“不合格”。
2) 对于不合格：必须指出缺失/不明确之处（对应到具体条款点），并给出可执行的补充建议。
3) 只输出结构化结果，不要输出多余字段，不要输出 Markdown。
4) 不要编造合同中不存在的信息；如果合同未提供某字段/信息，视为不满足。
"""


async def _read_upload_text(f: UploadFile) -> str:
    raw = await f.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{f.filename}' is empty.")
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        # 宽松兜底：有些文本可能是 gbk
        try:
            return raw.decode("gbk").strip()
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot decode '{f.filename}' as utf-8/gbk text.",
            ) from e


@app.post(
    "/v1/contract-check",
    response_model=ContractCheckResponse,
    response_model_by_alias=True,  # 输出中文字段：审查结果/审查过程/审查建议
)
async def contract_check(
    regulation: UploadFile = File(..., description="法规要求文本文件"),
    contract: UploadFile = File(..., description="合同内容文本文件"),
):
    regulation_text = await _read_upload_text(regulation)
    contract_text = await _read_upload_text(contract)

    chain = build_checker_chain()
    user_prompt = f"""请按法规要求审查合同内容，并输出结构化结果。

法规要求：
{regulation_text}

合同内容：
{contract_text}
"""

    try:
        result = chain.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        # with_structured_output 返回的通常是 Pydantic 对象
        if isinstance(result, ContractCheckResponse):
            return result
        return ContractCheckResponse.model_validate(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM contract-check failed: {type(e).__name__}: {e}") from e


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True})


