from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .schemas import ModelConfig
from .utils import estimate_tokens


@dataclass
class ModelResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: int
    raw: dict[str, Any]


class ModelClient(Protocol):
    config: ModelConfig

    async def check_health(self) -> dict[str, Any]:
        ...

    async def complete(self, prompt: str, system_prompt: str | None = None) -> ModelResult:
        ...


class OpenAICompatibleModelClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @property
    def api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key
        if self.config.api_key_env:
            return os.environ.get(self.config.api_key_env, "")
        return os.environ.get("OPENAI_API_KEY", "local-no-key")

    async def check_health(self) -> dict[str, Any]:
        if self.config.provider_type == "cloud_openai" and not self.api_key:
            return {
                "available": False,
                "detail": "缺少模型 API key",
            }

        url = self.config.base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=min(self.config.timeout, 10.0)) as client:
                response = await client.get(url, headers=headers)
            return {
                "available": 200 <= response.status_code < 300,
                "detail": f"GET /models -> HTTP {response.status_code}",
            }
        except Exception as exc:
            return {"available": False, "detail": str(exc)}

    async def complete(self, prompt: str, system_prompt: str | None = None) -> ModelResult:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_content = system_prompt or "You are a precise HLS C/C++ coding agent. Return only the requested patch contract."
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
        duration_ms = int((time.monotonic() - t0) * 1000)
        response.raise_for_status()
        raw = response.json()
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = raw.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(prompt))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
        return ModelResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            raw=raw,
        )


class LocalOpenAICompatibleModelClient(OpenAICompatibleModelClient):
    pass


class CloudOpenAICompatibleModelClient(OpenAICompatibleModelClient):
    pass


def build_model_client(config: ModelConfig) -> ModelClient:
    if config.provider_type == "cloud_openai":
        return CloudOpenAICompatibleModelClient(config)
    return LocalOpenAICompatibleModelClient(config)
