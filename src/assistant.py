"""
Claude-powered interview assistant.

Tracks conversation, detects interviewer questions, generates streaming answers
via Anthropic API with OAuth token.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Awaitable, Optional

import httpx

from src.config import (
    CLAUDE_API_URL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    get_anthropic_token,
)
from src.transcriber import TranscriptResult

logger = logging.getLogger(__name__)

# Max conversation turns to keep in context
_MAX_HISTORY_TURNS = 30


@dataclass
class ConversationTurn:
    speaker: str  # "Interviewer" or "You"
    text: str


class InterviewAssistant:
    """
    Listens to transcript events, maintains conversation history,
    and generates answers when the interviewer finishes speaking.
    """

    def __init__(
        self,
        system_prompt: str,
        debounce_seconds: float,
        on_answer_start: Callable[[], None],
        on_answer_chunk: Callable[[str], Awaitable[None]],
        on_answer_done: Callable[[], Awaitable[None]],
    ):
        self._system_prompt = system_prompt
        self._debounce_seconds = debounce_seconds
        self.on_answer_start = on_answer_start
        self.on_answer_chunk = on_answer_chunk
        self.on_answer_done = on_answer_done
        self._history: list[ConversationTurn] = []
        self._debounce_task: Optional[asyncio.Task] = None
        self._generating = False
        self._cancel_generation = asyncio.Event()
        self._http_client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._cancel_generation.set()
        await self._http_client.aclose()

    async def handle_transcript(self, result: TranscriptResult) -> None:
        """Called for each final transcript result."""
        if not result.is_final:
            return

        text = result.text.strip()
        if not text:
            return

        # Append to history
        self._history.append(ConversationTurn(speaker=result.speaker, text=text))

        # Trim history
        if len(self._history) > _MAX_HISTORY_TURNS:
            self._history = self._history[-_MAX_HISTORY_TURNS:]

        if result.speaker == "Interviewer":
            # Cancel any in-progress generation — question is still being asked
            if self._generating:
                self._cancel_generation.set()
            # Interviewer spoke — debounce before generating answer
            self._schedule_answer()

    def _schedule_answer(self) -> None:
        """Debounce: wait for interviewer to stop talking before generating."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounced_generate())

    async def _debounced_generate(self) -> None:
        try:
            await asyncio.sleep(self._debounce_seconds)
            await self._generate_answer()
        except asyncio.CancelledError:
            pass

    def _build_messages(self) -> list[dict]:
        """Convert conversation history to Claude API messages format."""
        messages = []
        # Group consecutive turns by role
        for turn in self._history:
            role = "assistant" if turn.speaker == "You" else "user"
            text = f"[{turn.speaker}]: {turn.text}"

            if messages and messages[-1]["role"] == role:
                # Merge consecutive same-role messages
                messages[-1]["content"] += f"\n{text}"
            else:
                messages.append({"role": role, "content": text})

        # Ensure messages start with user and alternate properly
        if messages and messages[0]["role"] == "assistant":
            messages.insert(0, {"role": "user", "content": "[Interview started]"})

        # Ensure last message is from user (interviewer) to prompt an answer
        if messages and messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "[Please provide or refine the answer based on the conversation above]"})

        return messages

    async def _generate_answer(self) -> None:
        """Stream an answer from Claude API."""
        messages = self._build_messages()
        if not messages:
            return

        self._generating = True
        self._cancel_generation.clear()

        token = get_anthropic_token()
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "system": self._system_prompt,
            "messages": messages,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
            "content-type": "application/json",
        }

        logger.info("Generating answer...")
        self.on_answer_start()

        try:
            async with self._http_client.stream(
                "POST", CLAUDE_API_URL, json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error("Claude API error %d: %s", response.status_code, body.decode())
                    return

                async for line in response.aiter_lines():
                    if self._cancel_generation.is_set():
                        logger.info("Answer generation cancelled (user speaking)")
                        break

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            await self.on_answer_chunk(text)

        except httpx.HTTPError as exc:
            logger.error("Claude API request failed: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Unexpected error during answer generation: %s", exc, exc_info=True)
        finally:
            self._generating = False
            await self.on_answer_done()
