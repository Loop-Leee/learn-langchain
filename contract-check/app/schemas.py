from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ReviewResult(str, Enum):
    qualified = "合格"
    unqualified = "不合格"


class ContractCheckResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    review_result: ReviewResult = Field(..., alias="审查结果", description="审查结果：合格/不合格")
    review_process: str = Field(
        ...,
        alias="审查过程",
        description="审查过程：指出合同与法规要求的对照依据与缺失点",
    )
    review_suggestion: str = Field(
        ...,
        alias="审查建议",
        description="审查建议：给出可执行的补充/修改建议",
    )


