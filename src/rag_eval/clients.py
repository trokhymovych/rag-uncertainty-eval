from __future__ import annotations


import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from anthropic import Anthropic
from openai import OpenAI

load_dotenv()


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


class OllamaProvider(BaseProvider):
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        # Normalise: native API lives at /api/chat, strip /v1 if passed
        self.base_url = base_url.rstrip("/").removesuffix("/v1")

    def generate(self, request: GenerationRequest) -> str:
        import json
        import urllib.request

        cfg = request.model_cfg
        payload: dict[str, Any] = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
        }
        if cfg.get("think") is not None:
            payload["think"] = cfg["think"]
        options: dict[str, Any] = {}
        if cfg.get("temperature") is not None:
            options["temperature"] = cfg["temperature"]
        if cfg.get("max_output_tokens") is not None:
            options["num_predict"] = int(cfg["max_output_tokens"])
        if options:
            payload["options"] = options

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return (result.get("message", {}).get("content") or "").strip()


class TogetherProvider(BaseProvider):
    def __init__(self, api_key_env: str = "TOGETHER_API_KEY") -> None:
        from together import Together

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set.")
        self.client = Together(api_key=api_key)

    def generate(self, request: GenerationRequest) -> str:
        cfg = request.model_cfg
        kwargs: dict[str, Any] = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        if cfg.get("temperature") is not None:
            kwargs["temperature"] = cfg["temperature"]
        if cfg.get("max_output_tokens") is not None:
            kwargs["max_tokens"] = int(cfg["max_output_tokens"])
        # Disable chain-of-thought reasoning for thinking-capable models
        if cfg.get("think") is False:
            kwargs["chat_template_kwargs"] = {"thinking": False}

        kwargs["stream"] = True
        stream = self.client.chat.completions.create(**kwargs)
        parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
        return "".join(parts).strip()


def make_provider(model_cfg: dict[str, Any]) -> BaseProvider:
    provider = model_cfg["provider"].lower()
    api_key_env = model_cfg.get("api_key_env")

    if provider == "openai":
        return OpenAIProvider(api_key_env or "OPENAI_API_KEY")
    if provider == "anthropic":
        return AnthropicProvider(api_key_env or "ANTHROPIC_API_KEY")
    if provider == "gemini":
        return GeminiProvider(api_key_env or "GEMINI_API_KEY")
    if provider == "ollama":
        return OllamaProvider(model_cfg.get("base_url", "http://localhost:11434/v1"))
    if provider == "together":
        return TogetherProvider(api_key_env or "TOGETHER_API_KEY")

    raise ValueError(f"Unsupported provider: {provider}")
