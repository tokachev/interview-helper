# Interview Helper

Real-time interview assistant: captures audio from a Zoom/Meet call, transcribes it via Deepgram, and generates answers using Claude API. You see the suggested answer in a markdown file while speaking.

## How it works

1. **BlackHole** captures system audio (interviewer's voice) as a virtual audio device
2. **Mic** captures your voice
3. **Deepgram** transcribes both streams in real-time via WebSocket
4. **Claude** generates an answer when the interviewer finishes a question
5. Two output files in `transcripts/`:
   - `answer.md` — current answer only (open in IDE/preview for live view)
   - `transcript_<timestamp>.md` — full conversation log

## Prerequisites

- macOS (tested on Sonoma+)
- Python 3.11+
- [BlackHole 2ch](https://existential.audio/blackhole/) — virtual audio driver

## Setup

### 1. Install BlackHole

Download and install from https://existential.audio/blackhole/

### 2. Configure Multi-Output Device

1. Open **Audio MIDI Setup** (`/Applications/Utilities/Audio MIDI Setup.app`)
2. Click `+` → **Create Multi-Output Device**
3. Check both your headphones/speakers AND BlackHole 2ch
4. Set this Multi-Output Device as your system output (System Settings → Sound → Output)

> **Note:** System volume slider will be disabled with Multi-Output Device. Use per-app volume controls instead.

### 3. Install Python dependencies

```bash
pip install deepgram-sdk sounddevice numpy httpx python-dotenv
```

### 4. Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

```
DEEPGRAM_API_KEY=<your key from console.deepgram.com>
ANTHROPIC_TOKEN=<your Claude API OAuth token>
DEEPGRAM_LANGUAGE=en
```

Supported languages: `en`, `ru`, `multi` (auto-detect, less reliable).

## Run

```bash
python3 -m src.main
```

Open `transcripts/answer.md` in your IDE with preview mode — it updates live as answers stream in.

Stop with `Ctrl+C`.

## Verify audio devices

```bash
python3 -c "from src.audio import print_devices; print_devices()"
```

Should show BlackHole and your mic detected. If BlackHole is missing, check that it's installed and your Multi-Output Device is configured.

## Troubleshooting

| Problem | Fix |
|---|---|
| BlackHole not found | Install BlackHole, restart, check Audio MIDI Setup |
| No transcription from interviewer | Verify Multi-Output Device is set as system output |
| Volume slider disabled | Expected with Multi-Output Device — use in-app volume |
| Bad recognition quality | Try explicit language (`DEEPGRAM_LANGUAGE=en`) instead of `multi` |
| Answer triggers too early on long questions | Increase `_DEBOUNCE_SECONDS` in `src/assistant.py` (default: 3s) |
