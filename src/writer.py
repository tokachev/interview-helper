"""
Transcript writer.

Two output files:
  - transcript.md: full conversation log (interviewer + you + assistant answers)
  - answer.md: ONLY the current assistant answer, overwritten each time — this is
    what you look at during the interview
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import TRANSCRIPT_DIR
from src.transcriber import TranscriptResult

logger = logging.getLogger(__name__)

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_COLOR_YOU = "\033[36m"
_COLOR_INTERVIEWER = "\033[33m"
_COLOR_ASSISTANT = "\033[32m"
_COLOR_INTERIM = "\033[2m"


class TranscriptWriter:

    def __init__(self):
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcript_path = TRANSCRIPT_DIR / f"transcript_{ts}.md"
        self.answer_path = TRANSCRIPT_DIR / "answer.md"
        self._transcript_file = None
        self._has_interim_on_screen = False

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
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        if self._transcript_file:
            self._transcript_file.flush()
            self._transcript_file.close()
            self._transcript_file = None

    # ---- Transcription events ----

    async def handle(self, result: TranscriptResult) -> None:
        text = result.text.strip()
        if not text:
            return

        color = _COLOR_YOU if result.speaker == "You" else _COLOR_INTERVIEWER

        if not result.is_final:
            line = f"{color}{_COLOR_INTERIM}[{result.speaker}] {text}...{_RESET}"
            sys.stdout.write(f"\r\033[K{line}")
            sys.stdout.flush()
            self._has_interim_on_screen = True
        else:
            if self._has_interim_on_screen:
                sys.stdout.write("\r\033[K")
                self._has_interim_on_screen = False

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"{_BOLD}{color}[{result.speaker}]{_RESET} {_COLOR_INTERIM}{ts}{_RESET} {text}")

            if self._transcript_file:
                self._transcript_file.write(f"**[{result.speaker}]** `{ts}` {text}\n")
                self._transcript_file.flush()

    # ---- AI answer streaming ----

    def start_answer_block(self) -> None:
        """Called before streaming starts. Clears the answer file."""
        if self._has_interim_on_screen:
            sys.stdout.write("\r\033[K")
            self._has_interim_on_screen = False

        ts = datetime.now().strftime("%H:%M:%S")

        # Terminal: header
        print(f"\n{_BOLD}{_COLOR_ASSISTANT}[Assistant]{_RESET} {_COLOR_INTERIM}{ts}{_RESET}")

        # Transcript file: header
        if self._transcript_file:
            self._transcript_file.write(f"\n**[Assistant]** `{ts}`\n\n")
            self._transcript_file.flush()

        # Answer file: overwrite with empty — will be filled chunk by chunk
        self.answer_path.write_text("", encoding="utf-8")

    async def handle_answer_chunk(self, text: str) -> None:
        """Append one streaming chunk to both outputs."""
        # Terminal
        sys.stdout.write(f"{_COLOR_ASSISTANT}{text}{_RESET}")
        sys.stdout.flush()

        # Transcript file
        if self._transcript_file:
            self._transcript_file.write(text)
            self._transcript_file.flush()

        # Answer file — append
        with open(self.answer_path, "a", encoding="utf-8") as f:
            f.write(text)

    async def handle_answer_done(self) -> None:
        """Called when answer streaming is complete."""
        print(f"\n{_COLOR_INTERIM}{'─' * 60}{_RESET}")

        if self._transcript_file:
            self._transcript_file.write("\n\n---\n\n")
            self._transcript_file.flush()
