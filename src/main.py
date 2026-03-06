"""
Interview helper — main entry point.

Starts two audio captures (mic + BlackHole) and two Deepgram connections,
routes transcriptions to the writer, and shuts down cleanly on Ctrl+C.

Usage:
    python -m src.main
    python -m src.main --profile backend
    python -m src.main --list-devices
    python -m src.main --list-profiles
"""

import argparse
import asyncio
import logging
import signal
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.assistant import InterviewAssistant
from src.audio import AudioCapture, find_blackhole_device, find_default_mic_device, list_devices
from src.config import get_anthropic_token, get_deepgram_api_key, DEEPGRAM_LANGUAGE
from src.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    get_profile_prompt,
    get_profile_title,
)
from src.transcriber import DeepgramTranscriber
from src.writer import TranscriptWriter

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="interview-helper",
        description="Real-time interview assistant powered by AI",
    )
    parser.add_argument(
        "--profile", "-p",
        default=DEFAULT_PROFILE,
        choices=sorted(PROFILES.keys()),
        help=f"Interview profile (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Override Deepgram language (e.g. 'ru', 'en', 'multi')",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=3.0,
        help="Seconds to wait after interviewer stops before generating (default: 3.0)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available interview profiles and exit",
    )
    return parser.parse_args()


def show_devices() -> None:
    """Pretty-print audio devices using Rich."""
    devices = list_devices()
    blackhole_idx = find_blackhole_device()
    mic_idx = find_default_mic_device()

    table = Table(title="Audio Devices", show_lines=False, padding=(0, 1))
    table.add_column("#", style="bold", justify="right", width=4)
    table.add_column("Name", min_width=30)
    table.add_column("In", justify="right", width=4)
    table.add_column("Out", justify="right", width=4)
    table.add_column("Rate", justify="right", width=7)
    table.add_column("Role", style="bold")

    for dev in devices:
        tags = []
        style = ""
        if dev["index"] == blackhole_idx:
            tags.append("[yellow]Interviewer[/yellow]")
            style = "yellow"
        if dev["index"] == mic_idx:
            tags.append("[cyan]Mic (You)[/cyan]")
            style = "cyan"

        table.add_row(
            str(dev["index"]),
            dev["name"],
            str(dev["max_input_channels"]),
            str(dev["max_output_channels"]),
            str(int(dev["default_samplerate"])),
            " ".join(tags),
            style=style if tags else "dim",
        )

    console.print()
    console.print(table)
    console.print()


def show_profiles() -> None:
    """Pretty-print available profiles."""
    table = Table(title="Interview Profiles", show_lines=True, padding=(0, 1))
    table.add_column("Name", style="bold cyan")
    table.add_column("Title")

    for name in sorted(PROFILES.keys()):
        table.add_row(name, PROFILES[name]["title"])

    console.print()
    console.print(table)
    console.print()
    console.print(f"  Default: [bold]{DEFAULT_PROFILE}[/bold]")
    console.print(f"  Usage:   [dim]python -m src.main --profile backend[/dim]")
    console.print()


def show_startup_banner(args: argparse.Namespace, mic_idx, blackhole_idx) -> None:
    """Display a nice startup panel with current configuration."""
    devices = list_devices()

    mic_name = devices[mic_idx]["name"] if mic_idx is not None else "Not found"
    bh_name = (
        devices[blackhole_idx]["name"]
        if blackhole_idx is not None
        else "[red]Not found[/red]"
    )

    info = Text()
    info.append("Profile:      ", style="dim")
    info.append(f"{get_profile_title(args.profile)}", style="bold cyan")
    info.append(f" ({args.profile})\n", style="dim")
    info.append("Mic:          ", style="dim")
    info.append(f"{mic_name}\n", style="cyan")
    info.append("Interviewer:  ", style="dim")
    info.append(f"{bh_name}\n", style="yellow")
    info.append("Language:     ", style="dim")
    info.append(f"{args.language or DEEPGRAM_LANGUAGE}\n", style="white")
    info.append("Debounce:     ", style="dim")
    info.append(f"{args.debounce}s\n", style="white")

    panel = Panel(
        info,
        title="[bold]Interview Helper[/bold]",
        subtitle="[dim]Ctrl+C to stop[/dim]",
        border_style="green",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


async def main(args: argparse.Namespace) -> None:
    # -----------------------------------------------------------------------
    # 1. Validate config early
    # -----------------------------------------------------------------------
    for check in (get_deepgram_api_key, get_anthropic_token):
        try:
            check()
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Resolve audio devices
    # -----------------------------------------------------------------------
    mic_idx = find_default_mic_device()
    blackhole_idx = find_blackhole_device()

    if mic_idx is None:
        console.print("[bold red]Error:[/bold red] No microphone found.")
        sys.exit(1)

    if blackhole_idx is None:
        console.print(
            "[yellow]Warning:[/yellow] BlackHole not found — only mic stream will be active.\n"
            "  Install BlackHole from [link]https://existential.audio[/link] for interviewer audio."
        )

    # -----------------------------------------------------------------------
    # 3. Show startup banner
    # -----------------------------------------------------------------------
    show_startup_banner(args, mic_idx, blackhole_idx)

    # -----------------------------------------------------------------------
    # 4. Resolve system prompt from profile
    # -----------------------------------------------------------------------
    system_prompt = get_profile_prompt(args.profile)

    # Override language if specified
    if args.language:
        from src import config
        config.DEEPGRAM_OPTIONS["language"] = args.language

    # -----------------------------------------------------------------------
    # 5. Set up writer + assistant
    # -----------------------------------------------------------------------
    writer = TranscriptWriter()
    writer.open()
    console.print(f"  [dim]Transcript:[/dim] {writer.transcript_path}")
    console.print(f"  [dim]Live answer:[/dim] {writer.answer_path}")
    console.print()

    assistant = InterviewAssistant(
        system_prompt=system_prompt,
        debounce_seconds=args.debounce,
        on_answer_start=writer.start_answer_block,
        on_answer_chunk=writer.handle_answer_chunk,
        on_answer_done=writer.handle_answer_done,
    )

    # -----------------------------------------------------------------------
    # 6. Set up audio queues and captures
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
    # 7. Build transcriber coroutines
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
    # 8. Start audio streams then transcribers
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

    console.print("[bold green]Recording started.[/bold green] Listening...\n")

    # -----------------------------------------------------------------------
    # 9. Wait until cancelled
    # -----------------------------------------------------------------------
    shutdown_event = asyncio.Event()

    def _handle_signal() -> None:
        console.print("\n[yellow]Shutting down...[/yellow]")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await shutdown_event.wait()

    # -----------------------------------------------------------------------
    # 10. Graceful shutdown
    # -----------------------------------------------------------------------
    mic_capture.stop()
    if bh_capture is not None:
        bh_capture.stop()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    await assistant.close()
    writer.close()

    console.print(f"\n[green]Transcript saved:[/green] {writer.transcript_path}")


def cli_entry() -> None:
    """Entry point for the CLI (pyproject.toml console_scripts)."""
    args = parse_args()

    if args.list_devices:
        show_devices()
        return

    if args.list_profiles:
        show_profiles()
        return

    asyncio.run(main(args))


if __name__ == "__main__":
    cli_entry()
