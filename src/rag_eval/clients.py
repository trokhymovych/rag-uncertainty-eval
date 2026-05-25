from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from google import genai
from anthropic import Anthropic
from openai import OpenAI


@dataclass
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    model_cfg: dict[str, Any]


class BaseProvider:
    def generate(self, request: GenerationRequest) -> str:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key_env: str = "OPENAI_API_KEY") -> None:

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set.")
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _extract_text(response) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        chunks = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []):
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(text)
        return "\n".join(chunks).strip()

    def generate(self, request: GenerationRequest) -> str:
        cfg = request.model_cfg
        params: dict[str, Any] = {
            "model": cfg["model"],
            "input": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        if cfg.get("temperature") is not None:
            params["temperature"] = cfg["temperature"]
        if cfg.get("max_output_tokens") is not None:
            params["max_output_tokens"] = cfg["max_output_tokens"]

        response = self.client.responses.create(**params)
        return self._extract_text(response)


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY") -> None:

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set.")
        self.client = Anthropic(api_key=api_key)

    @staticmethod
    def _extract_text(response) -> str:
        chunks = []
        for item in getattr(response, "content", []):
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks).strip()

    def generate(self, request: GenerationRequest) -> str:
        cfg = request.model_cfg
        response = self.client.messages.create(
            model=cfg["model"],
            system=request.system_prompt,
            messages=[{"role": "user", "content": request.user_prompt}],
            temperature=cfg.get("temperature", 0),
            max_tokens=int(cfg.get("max_output_tokens") or 128),
        )
        return self._extract_text(response)


class GeminiProvider(BaseProvider):
    def __init__(self, api_key_env: str = "GEMINI_API_KEY") -> None:

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set.")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _extract_text(response) -> str:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return str(response)

    def generate(self, request: GenerationRequest) -> str:
        from google.genai import types

        cfg = request.model_cfg
        gen_cfg = types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=cfg.get("temperature", 0),
            max_output_tokens=int(cfg.get("max_output_tokens") or 128),
        )
        response = self.client.models.generate_content(
            model=cfg["model"],
            contents=request.user_prompt,
            config=gen_cfg,
        )
        return self._extract_text(response)


def make_provider(model_cfg: dict[str, Any]) -> BaseProvider:
    provider = model_cfg["provider"].lower()
    api_key_env = model_cfg.get("api_key_env")

    if provider == "openai":
        return OpenAIProvider(api_key_env or "OPENAI_API_KEY")
    if provider == "anthropic":
        return AnthropicProvider(api_key_env or "ANTHROPIC_API_KEY")
    if provider == "gemini":
        return GeminiProvider(api_key_env or "GEMINI_API_KEY")

    raise ValueError(f"Unsupported provider: {provider}")
