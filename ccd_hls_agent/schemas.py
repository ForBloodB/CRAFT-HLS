from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Stage(StrEnum):
    INIT = "INIT"
    STATIC_SCAN = "STATIC_SCAN"
    CONTEXT_ATOMIZE = "CONTEXT_ATOMIZE"
    CONTEXT_SCORE = "CONTEXT_SCORE"
    RAG_RETRIEVE = "RAG_RETRIEVE"
    PROMPT_BUILD = "PROMPT_BUILD"
    LLM_CALL = "LLM_CALL"
    PATCH_VALIDATE = "PATCH_VALIDATE"
    HLS_EVAL_RUN = "HLS_EVAL_RUN"
    MEMORY_WRITE = "MEMORY_WRITE"
    NEXT_ITERATION = "NEXT_ITERATION"
    DONE = "DONE"


class ModelConfig(BaseModel):
    profile_name: str = "local_qwen_coder"
    provider_type: Literal["local_openai", "cloud_openai"] = "local_openai"
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    api_key_env: str | None = None
    model: str = "Qwen3-Coder-30B-A3B-Instruct"
    temperature: float = 0.2
    max_tokens: int = 2048
    context_window: int = 16384
    timeout: float = 60.0


class AtomRecord(BaseModel):
    atom_id: str
    task_id: str
    run_id: str
    kind: str
    scope: str
    stage: str
    summary: str
    evidence_uri: str | None = None
    code_hash: str | None = None
    status: str = "active"
    token_estimate: int = 1
    value_score: float = 0.0
    certainty_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
