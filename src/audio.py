"""
Audio device discovery and capture.

Two roles:
  - Mic    → speaker label "You"
  - BlackHole → speaker label "Interviewer" (captures system audio via virtual device)
"""

import asyncio
import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from src.config import (
    BLACKHOLE_NAME_FRAGMENT,
    CHANNELS,
    CHUNK_FRAMES,
    DTYPE,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def list_devices() -> list[dict]:
    """Return all audio devices as a list of dicts."""
    devices = sd.query_devices()
    result = []
    for idx, dev in enumerate(devices):
        result.append({
            "index": idx,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        })
    return result


def find_blackhole_device() -> Optional[int]:
    """
    Return the device index of the BlackHole virtual audio device, or None.
    Matches by case-insensitive substring of the device name.
    """
    for dev in list_devices():
        if (
            BLACKHOLE_NAME_FRAGMENT.lower() in dev["name"].lower()
            and dev["max_input_channels"] > 0
        ):
            return dev["index"]
    return None


def find_default_mic_device() -> Optional[int]:
    """
    Return the device index of the system default input device, or None.
    Excludes BlackHole and virtual devices.
    """
    try:
        default_input = sd.default.device[0]
        if default_input is not None and default_input >= 0:
            return int(default_input)
    except Exception:
        pass

    # Fallback: first real input device that isn't BlackHole
    for dev in list_devices():
        if (
            dev["max_input_channels"] > 0
            and BLACKHOLE_NAME_FRAGMENT.lower() not in dev["name"].lower()
        ):
            return dev["index"]
    return None


def print_devices() -> None:
    """Pretty-print all audio devices and identify BlackHole + default mic."""
    devices = list_devices()
    blackhole_idx = find_blackhole_device()
    mic_idx = find_default_mic_device()

    print(f"\n{'idx':>4}  {'name':<45}  {'in':>3}  {'out':>3}  {'rate':>7}")
    print("-" * 70)
    for dev in devices:
        tags = []
        if dev["index"] == blackhole_idx:
            tags.append("BlackHole")
        if dev["index"] == mic_idx:
            tags.append("Mic (default)")
        tag_str = f"  <- {', '.join(tags)}" if tags else ""
        print(
            f"{dev['index']:>4}  {dev['name']:<45}  "
            f"{dev['max_input_channels']:>3}  {dev['max_output_channels']:>3}  "
            f"{int(dev['default_samplerate']):>7}{tag_str}"
        )

    print()
    if blackhole_idx is not None:
        print(f"BlackHole device found: index {blackhole_idx}")
    else:
        print("WARNING: BlackHole device NOT found. Install BlackHole from existential.audio.")

    if mic_idx is not None:
        mic_name = devices[mic_idx]["name"]
        print(f"Default mic device: index {mic_idx} ({mic_name})")
    else:
        print("WARNING: No input device found.")
    print()


# ---------------------------------------------------------------------------
# Async audio capture
# ---------------------------------------------------------------------------

class AudioCapture:
    """
    Captures audio from a single device and pushes raw int16 bytes to an asyncio.Queue.

    Usage:
        capture = AudioCapture(device_index=3, queue=my_queue, loop=asyncio.get_event_loop())
        capture.start()
        ...
        capture.stop()
    """

    def __init__(
        self,
        device_index: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        label: str = "unknown",
    ):
        self.device_index = device_index
        self.queue = queue
        self.loop = loop
        self.label = label
        self._stream: Optional[sd.RawInputStream] = None
        self._stop_event = threading.Event()

    def _callback(self, indata: bytes, frames: int, time_info, status) -> None:
        if status:
            logger.warning("[%s] sounddevice status: %s", self.label, status)
        # indata is a memoryview of int16 samples; copy to bytes before queuing
        data = bytes(indata)
        # Thread-safe enqueue to asyncio queue from the sounddevice callback thread
        asyncio.run_coroutine_threadsafe(self.queue.put(data), self.loop)

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError(f"AudioCapture [{self.label}] already started")

        logger.info(
            "[%s] Opening input stream on device %d @ %dHz, %d ch, %s",
            self.label,
            self.device_index,
            SAMPLE_RATE,
            CHANNELS,
            DTYPE,
        )
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_FRAMES,
            device=self.device_index,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("[%s] Audio capture started", self.label)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("[%s] Audio capture stopped", self.label)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_devices()
