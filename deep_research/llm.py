"""LLM client with Groq primary + Gemini fallback + local Ollama fallback.

- `chat_stream(...)` yields text deltas asynchronously.
- `chat_json(...)` returns a single parsed JSON object (used for the planner).
- Auto-failover order: Groq → Gemini → local Ollama. Each tier is skipped if
  unconfigured or unreachable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional

import httpx

from deep_research.models import Message

log = logging.getLogger(__name__)

# Lazy imports so the module loads even if optional providers are missing.


def _approx_tokens(text: str) -> int:
    """Cheap token estimator used when tiktoken isn't decisive (~4 chars / token)."""
    return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    """Best-effort token count. Uses tiktoken cl100k_base if available."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return _approx_tokens(text)


class LLMError(Exception):
    pass


class LLMClient:
    """Async chat client with provider failover."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        gemini_model: str = "gemini-2.5-flash",
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> None:
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", groq_model)
        self.gemini_model = os.getenv("GEMINI_MODEL", gemini_model)

        self.ollama_url = (ollama_url or os.getenv("OLLAMA_URL", "")).rstrip("/")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        # Ollama is "configured" if a URL was given (defaults to localhost when not set).
        # We probe lazily at first use and silently disable it on connection failure.
        if not self.ollama_url:
            self.ollama_url = "http://localhost:11434"
        self._ollama_available: Optional[bool] = None  # tri-state: unknown / True / False

        self._groq = None
        self._gemini = None

        if not (self.groq_api_key or self.gemini_api_key or self.ollama_url):
            raise LLMError(
                "No LLM available. Configure at least one of: "
                "GROQ_API_KEY, GEMINI_API_KEY, OLLAMA_URL (with `ollama serve` running)."
            )

    # ------------------------------------------------------------------ lazy clients

    def _get_groq(self):
        if self._groq is None and self.groq_api_key:
            from groq import AsyncGroq  # type: ignore

            self._groq = AsyncGroq(api_key=self.groq_api_key)
        return self._groq

    def _get_gemini(self):
        if self._gemini is None and self.gemini_api_key:
            from google import genai  # type: ignore

            self._gemini = genai.Client(api_key=self.gemini_api_key)
        return self._gemini

    async def _probe_ollama(self) -> bool:
        """Quick reachability check; cached after the first probe."""
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
            self._ollama_available = resp.status_code == 200
        except Exception:  # noqa: BLE001
            self._ollama_available = False
        if not self._ollama_available:
            log.info("Ollama unreachable at %s — fallback disabled", self.ollama_url)
        return self._ollama_available

    # ------------------------------------------------------------------ messages helpers

    @staticmethod
    def _to_openai_messages(messages: list[Message] | list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if isinstance(m, Message):
                out.append({"role": m.role, "content": m.content})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    @staticmethod
    def _to_gemini_contents(messages: list[Message] | list[dict]) -> tuple[str, list[dict]]:
        """Gemini wants a system_instruction string + alternating user/model contents."""
        sys_parts: list[str] = []
        contents: list[dict] = []
        for m in messages:
            role = m.role if isinstance(m, Message) else m["role"]
            content = m.content if isinstance(m, Message) else m["content"]
            if role == "system":
                sys_parts.append(content)
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
        return ("\n\n".join(sys_parts).strip(), contents)

    # ------------------------------------------------------------------ streaming chat

    async def chat_stream(
        self,
        messages: list[Message] | list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Tries Groq → Gemini → Ollama in order, falling back on failure."""
        last_err: Optional[Exception] = None

        if self._get_groq():
            try:
                async for delta in self._groq_stream(messages, temperature, max_tokens):
                    yield delta
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("Groq streaming failed (%s); falling back", e)

        if self._get_gemini():
            try:
                async for delta in self._gemini_stream(messages, temperature, max_tokens):
                    yield delta
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("Gemini streaming failed (%s); falling back", e)

        if await self._probe_ollama():
            try:
                async for delta in self._ollama_stream(messages, temperature, max_tokens):
                    yield delta
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("Ollama streaming failed: %s", e)

        raise LLMError(f"All LLM providers failed: {last_err!s}")

    async def _groq_stream(
        self,
        messages,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = self._get_groq()
        stream = await client.chat.completions.create(  # type: ignore[union-attr]
            model=self.groq_model,
            messages=self._to_openai_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except (AttributeError, IndexError):
                delta = ""
            if delta:
                yield delta

    async def _gemini_stream(
        self,
        messages,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = self._get_gemini()
        sys_instruction, contents = self._to_gemini_contents(messages)
        config: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if sys_instruction:
            config["system_instruction"] = sys_instruction
        stream = await client.aio.models.generate_content_stream(  # type: ignore[union-attr]
            model=self.gemini_model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            txt = getattr(chunk, "text", "") or ""
            if txt:
                yield txt

    async def _ollama_stream(
        self,
        messages,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model": self.ollama_model,
            "messages": self._to_openai_messages(messages),
            "stream": True,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if json_mode:
            payload["format"] = "json"
        url = f"{self.ollama_url}/api/chat"
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=2.0)) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise LLMError(f"Ollama HTTP {resp.status_code}: {body[:300]}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (obj.get("message") or {}).get("content") or ""
                    if delta:
                        yield delta
                    if obj.get("done"):
                        break

    # ------------------------------------------------------------------ JSON (planner)

    async def chat_json(
        self,
        messages: list[Message] | list[dict],
        temperature: float = 0.1,
        max_tokens: int = 800,
        max_retries: int = 1,
    ) -> dict:
        """Force a JSON response; tolerate models that emit fenced code. Tries Groq → Gemini → Ollama."""
        last_err: Optional[Exception] = None
        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            try:
                if self._get_groq():
                    raw = await self._groq_json(messages, temperature, max_tokens)
                elif self._get_gemini():
                    raw = await self._gemini_json(messages, temperature, max_tokens)
                elif await self._probe_ollama():
                    raw = await self._ollama_json(messages, temperature, max_tokens)
                else:
                    raise LLMError("No LLM provider available")
                return _safe_json_loads(raw)
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("chat_json attempt %d failed: %s", attempt, e)
                await asyncio.sleep(0.5)
        raise LLMError(f"chat_json failed after {max_retries + 1} attempts: {last_err!s}")

    async def _groq_json(self, messages, temperature: float, max_tokens: int) -> str:
        client = self._get_groq()
        resp = await client.chat.completions.create(  # type: ignore[union-attr]
            model=self.groq_model,
            messages=self._to_openai_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or "{}"

    async def _gemini_json(self, messages, temperature: float, max_tokens: int) -> str:
        client = self._get_gemini()
        sys_instruction, contents = self._to_gemini_contents(messages)
        config: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        }
        if sys_instruction:
            config["system_instruction"] = sys_instruction
        resp = await client.aio.models.generate_content(  # type: ignore[union-attr]
            model=self.gemini_model,
            contents=contents,
            config=config,
        )
        return getattr(resp, "text", "") or "{}"

    async def _ollama_json(self, messages, temperature: float, max_tokens: int) -> str:
        parts: list[str] = []
        async for d in self._ollama_stream(messages, temperature, max_tokens, json_mode=True):
            parts.append(d)
        return "".join(parts) or "{}"

    # ------------------------------------------------------------------ non-streaming chat

    async def chat(
        self,
        messages: list[Message] | list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        """Collect a streamed response into a single string."""
        out: list[str] = []
        async for delta in self.chat_stream(messages, temperature, max_tokens):
            out.append(delta)
        return "".join(out)


def _safe_json_loads(text: str) -> dict:
    """Parse JSON tolerating ```json fences and stray prefixes."""
    s = (text or "").strip()
    if s.startswith("```"):
        # strip first fence line and the trailing fence
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
        # if there is a language hint left over (e.g. 'json\n{...}')
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # If the model produced extra prose around the JSON, grab the outermost {...}.
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            s = s[start : end + 1]
    return json.loads(s)
