"""
Deepgram live transcription client (deepgram-sdk v6).

One DeepgramTranscriber per audio stream (mic or BlackHole). Connects via
WebSocket, streams raw int16 PCM from an asyncio.Queue, parses transcript
results, and calls the provided async callback for each non-empty result.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from src.config import DEEPGRAM_OPTIONS, SAMPLE_RATE, get_deepgram_api_key

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    speaker: str            # "You" or "Interviewer"
    text: str
    is_final: bool          # Deepgram is_final flag
    speech_final: bool      # Deepgram speech_final flag (endpointing)
    language: Optional[str] # Detected language code, if available


# Async callback that receives a TranscriptResult
TranscriptCallback = Callable[[TranscriptResult], Awaitable[None]]

# Keepalive interval in seconds — Deepgram closes idle connections after ~10s
_KEEPALIVE_INTERVAL = 8.0


class DeepgramTranscriber:
    """
    Manages one Deepgram live transcription WebSocket connection.

    Args:
        speaker:      Label for transcript lines ("You" or "Interviewer").
        audio_queue:  asyncio.Queue fed by AudioCapture with raw int16 bytes.
        on_transcript: Async callback for each non-empty transcript segment.
    """

    def __init__(
        self,
        speaker: str,
        audio_queue: asyncio.Queue,
        on_transcript: TranscriptCallback,
    ):
        self.speaker = speaker
        self.audio_queue = audio_queue
        self.on_transcript = on_transcript

    async def run(self) -> None:
        """Open Deepgram WebSocket and stream audio until cancelled."""
        api_key = get_deepgram_api_key()
        client = AsyncDeepgramClient(api_key=api_key)

        connect_kwargs = dict(
            model=DEEPGRAM_OPTIONS["model"],
            language=DEEPGRAM_OPTIONS["language"],
            punctuate=str(DEEPGRAM_OPTIONS["punctuate"]).lower(),
            interim_results=str(DEEPGRAM_OPTIONS["interim_results"]).lower(),
            endpointing=str(DEEPGRAM_OPTIONS["endpointing"]),
            encoding="linear16",
            sample_rate=str(SAMPLE_RATE),
            channels="1",
        )

        logger.info("[%s] Connecting to Deepgram WebSocket...", self.speaker)

        try:
            async with client.listen.v1.connect(**connect_kwargs) as ws:
                logger.info("[%s] Deepgram connection established", self.speaker)

                # Run sender and receiver concurrently
                send_task = asyncio.create_task(
                    self._send_loop(ws), name=f"dg-send-{self.speaker}"
                )
                recv_task = asyncio.create_task(
                    self._recv_loop(ws), name=f"dg-recv-{self.speaker}"
                )

                # Wait until one of them finishes (cancelled or error)
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

                # Re-raise if either task raised an unhandled exception
                for t in done:
                    if not t.cancelled() and t.exception():
                        logger.error(
                            "[%s] Transcriber task error: %s",
                            self.speaker,
                            t.exception(),
                        )

        except asyncio.CancelledError:
            logger.info("[%s] Transcriber cancelled", self.speaker)
            raise
        except Exception as exc:
            logger.error("[%s] Deepgram connection error: %s", self.speaker, exc, exc_info=True)
        finally:
            logger.info("[%s] Deepgram connection closed", self.speaker)

    async def _send_loop(self, ws) -> None:
        """Pull audio chunks from the queue and send to Deepgram."""
        last_send = asyncio.get_event_loop().time()

        while True:
            try:
                chunk: bytes = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                try:
                    await ws.send_media(chunk)
                except Exception:
                    # Connection closed (e.g. during shutdown) — exit gracefully
                    return
                last_send = asyncio.get_event_loop().time()
            except asyncio.TimeoutError:
                # No audio in the last second — send keepalive if needed
                now = asyncio.get_event_loop().time()
                if now - last_send >= _KEEPALIVE_INTERVAL:
                    try:
                        await ws.send_keep_alive()
                        last_send = now
                    except Exception as exc:
                        logger.warning("[%s] Keepalive failed: %s", self.speaker, exc)
            except asyncio.CancelledError:
                # Send CloseStream before exiting so Deepgram flushes its buffer
                try:
                    await ws.send_close_stream()
                except Exception:
                    pass
                raise

    async def _recv_loop(self, ws) -> None:
        """Receive transcript events from Deepgram and invoke the callback."""
        async for message in ws:
            if not isinstance(message, ListenV1Results):
                # Metadata, UtteranceEnd, SpeechStarted, etc. — skip
                continue

            alternatives = message.channel.alternatives
            if not alternatives:
                continue

            text = alternatives[0].transcript
            if not text or not text.strip():
                continue

            # Language is in the metadata, not per-channel in v6
            detected_lang: Optional[str] = None
            try:
                detected_lang = message.metadata.extra.get("detected_language")  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                pass

            result = TranscriptResult(
                speaker=self.speaker,
                text=text.strip(),
                is_final=bool(message.is_final),
                speech_final=bool(message.speech_final),
                language=detected_lang,
            )
            try:
                await self.on_transcript(result)
            except Exception as exc:
                logger.error(
                    "[%s] on_transcript callback error: %s", self.speaker, exc, exc_info=True
                )
