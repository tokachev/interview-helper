"""
Transcript writer with Rich terminal UI.

Two output files:
  - transcript.md: full conversation log (interviewer + you + assistant answers)
  - answer.md: ONLY the current assistant answer, overwritten each time
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.config import TRANSCRIPT_DIR
from src.transcriber import TranscriptResult

logger = logging.getLogger(__name__)

console = Console()

# Speaker styles
_STYLE_YOU = "bold cyan"
_STYLE_INTERVIEWER = "bold yellow"
_STYLE_ASSISTANT = "bold green"
_STYLE_TIMESTAMP = "dim"
_STYLE_INTERIM = "dim italic"


class TranscriptWriter:

    def __init__(self):
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcript_path = TRANSCRIPT_DIR / f"transcript_{ts}.md"
        self.answer_path = TRANSCRIPT_DIR / "answer.md"
        self._transcript_file = None
        self._answer_file: Optional[object] = None
        self._has_interim_on_screen = False
        self._current_answer_text = ""

    def open(self) -> None:
        self._transcript_file = open(self.transcript_path, "a", encoding="utf-8")
        self._transcript_file.write(
            f"\n# Interview session — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        self._transcript_file.flush()

        # Clear the answer file
        self.answer_path.write_text("*Waiting for interviewer question...*\n", encoding="utf-8")

        logger.info("Transcript: %s", self.transcript_path)
        logger.info("Answer file: %s", self.answer_path)

    def close(self) -> None:
        if self._has_interim_on_screen:
            console.print()  # newline to clear interim
        if self._answer_file:
            self._answer_file.close()
            self._answer_file = None
        if self._transcript_file:
            self._transcript_file.flush()
            self._transcript_file.close()
            self._transcript_file = None

    # ---- Transcription events ----

    async def handle(self, result: TranscriptResult) -> None:
        text = result.text.strip()
        if not text:
            return

        style = _STYLE_YOU if result.speaker == "You" else _STYLE_INTERVIEWER

        if not result.is_final:
            # Interim result — show dimmed, overwrite previous
            line = Text()
            line.append(f"  ⟩ [{result.speaker}] ", style=_STYLE_INTERIM)
            line.append(f"{text}...", style=_STYLE_INTERIM)
            console.print(line, end="\r", overflow="ellipsis")
            self._has_interim_on_screen = True
        else:
            if self._has_interim_on_screen:
                console.print(" " * 120, end="\r")  # clear interim line
                self._has_interim_on_screen = False

            ts = datetime.now().strftime("%H:%M:%S")
            line = Text()
            line.append(f"  [{result.speaker}]", style=style)
            line.append(f" {ts} ", style=_STYLE_TIMESTAMP)
            line.append(text)
            console.print(line)

            if self._transcript_file:
                self._transcript_file.write(f"**[{result.speaker}]** `{ts}` {text}\n")
                self._transcript_file.flush()

    # ---- AI answer streaming ----

    def start_answer_block(self) -> None:
        """Called before streaming starts. Clears the answer file."""
        if self._has_interim_on_screen:
            console.print(" " * 120, end="\r")
            self._has_interim_on_screen = False

        ts = datetime.now().strftime("%H:%M:%S")

        # Terminal: divider + header
        console.print()
        console.rule("[bold green]Assistant", style="green", align="left")

        # Transcript file: header
        if self._transcript_file:
            self._transcript_file.write(f"\n**[Assistant]** `{ts}`\n\n")
            self._transcript_file.flush()

        # Answer file: open and keep open for the duration of streaming
        if self._answer_file:
            self._answer_file.close()
        self._answer_file = open(self.answer_path, "w", encoding="utf-8")
        self._current_answer_text = ""

    async def handle_answer_chunk(self, text: str) -> None:
        """Append one streaming chunk to both outputs."""
        self._current_answer_text += text

        # Terminal — print chunk inline
        console.print(text, end="", style="green", highlight=False)

        # Transcript file
        if self._transcript_file:
            self._transcript_file.write(text)
            self._transcript_file.flush()

        # Answer file — write and flush (file kept open)
        if self._answer_file:
            self._answer_file.write(text)
            self._answer_file.flush()

    async def handle_answer_done(self) -> None:
        """Called when answer streaming is complete."""
        console.print()  # newline after streamed text
        console.rule(style="dim")
        console.print()

        if self._transcript_file:
            self._transcript_file.write("\n\n---\n\n")
            self._transcript_file.flush()

        # Close answer file
        if self._answer_file:
            self._answer_file.close()
            self._answer_file = None
