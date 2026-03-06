"""
Interview helper — main entry point.

Starts two audio captures (mic + BlackHole) and two Deepgram connections,
routes transcriptions to the writer, and shuts down cleanly on Ctrl+C.

Usage:
    python -m src.main
"""

import asyncio
import logging
import signal
import sys

from src.assistant import InterviewAssistant
from src.audio import AudioCapture, find_blackhole_device, find_default_mic_device
from src.config import get_anthropic_token, get_deepgram_api_key
from src.transcriber import DeepgramTranscriber
from src.writer import TranscriptWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Validate config early
    # -----------------------------------------------------------------------
    for check in (get_deepgram_api_key, get_anthropic_token):
        try:
            check()
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Resolve audio devices
    # -----------------------------------------------------------------------
    mic_idx = find_default_mic_device()
    blackhole_idx = find_blackhole_device()

    if mic_idx is None:
        logger.error("No microphone input device found. Cannot start.")
        sys.exit(1)

    if blackhole_idx is None:
        logger.warning(
            "BlackHole device not found — only mic (You) stream will be active. "
            "Install BlackHole from existential.audio for interviewer audio."
        )

    logger.info("Mic device index: %s", mic_idx)
    logger.info("BlackHole device index: %s", blackhole_idx)

    # -----------------------------------------------------------------------
    # 3. Set up writer + assistant
    # -----------------------------------------------------------------------
    writer = TranscriptWriter()
    writer.open()
    logger.info("Open answer file for live view: %s", writer.answer_path)

    assistant = InterviewAssistant(
        on_answer_start=writer.start_answer_block,
        on_answer_chunk=writer.handle_answer_chunk,
        on_answer_done=writer.handle_answer_done,
    )

    # -----------------------------------------------------------------------
    # 4. Set up audio queues and captures
    # -----------------------------------------------------------------------
    loop = asyncio.get_running_loop()

    mic_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    mic_capture = AudioCapture(
        device_index=mic_idx,
        queue=mic_queue,
        loop=loop,
        label="You",
    )

    bh_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    bh_capture: AudioCapture | None = None
    if blackhole_idx is not None:
        bh_capture = AudioCapture(
            device_index=blackhole_idx,
            queue=bh_queue,
            loop=loop,
            label="Interviewer",
        )

    # -----------------------------------------------------------------------
    # 5. Build transcriber coroutines
    # -----------------------------------------------------------------------
    async def on_transcript(result):
        """Fan out transcript to writer + assistant."""
        await writer.handle(result)
        await assistant.handle_transcript(result)

    mic_transcriber = DeepgramTranscriber(
        speaker="You",
        audio_queue=mic_queue,
        on_transcript=on_transcript,
    )

    tasks: list[asyncio.Task] = []

    # -----------------------------------------------------------------------
    # 6. Start audio streams then transcribers
    # -----------------------------------------------------------------------
    mic_capture.start()
    if bh_capture is not None:
        bh_capture.start()

    tasks.append(asyncio.create_task(mic_transcriber.run(), name="transcriber-you"))

    if bh_capture is not None:
        bh_transcriber = DeepgramTranscriber(
            speaker="Interviewer",
            audio_queue=bh_queue,
            on_transcript=on_transcript,
        )
        tasks.append(
            asyncio.create_task(bh_transcriber.run(), name="transcriber-interviewer")
        )

    logger.info("Recording started. Press Ctrl+C to stop.\n")

    # -----------------------------------------------------------------------
    # 7. Wait until cancelled
    # -----------------------------------------------------------------------
    shutdown_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("\nShutdown signal received...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await shutdown_event.wait()

    # -----------------------------------------------------------------------
    # 8. Graceful shutdown
    # -----------------------------------------------------------------------
    logger.info("Stopping audio captures...")
    mic_capture.stop()
    if bh_capture is not None:
        bh_capture.stop()

    logger.info("Cancelling transcriber tasks...")
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    await assistant.close()
    writer.close()
    logger.info("Goodbye. Transcript saved to: %s", writer.transcript_path)


if __name__ == "__main__":
    asyncio.run(main())
