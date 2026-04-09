# CESA-Chess

**CESA (Cognitive Embodied Strategic Agent)** is an AI-powered robotic chess system that integrates a Stockfish chess engine, Google Gemini LLM, and an ABB YuMi robot arm to physically play chess on a real board — with computer vision for move detection and natural language interaction.

---

## Overview

CESA bridges the gap between digital chess AI and physical robotics. When CESA decides on a move, it does not just update a board on screen — it picks up a chess piece with a robot arm and places it on the correct square. You can talk to CESA before and during the game, ask for chess advice, or issue robot commands in plain language.

```
You  ──► speaks/types to CESA
          │
          ▼
    Gemini LLM (CESA)  ──► chess strategy, natural conversation
          │
          ├──► Stockfish Engine  ──► calculates best move
          │
          ├──► Computer Vision (YOLO)  ──► detects your physical moves
          │
          └──► ABB YuMi Robot Arm  ──► physically moves pieces on the board
```

---

## Features

- **Physical chess gameplay** — ABB YuMi robot arm moves pieces on a real chessboard
- **Computer vision** — YOLO-based detection reads your physical moves via a camera feed
- **Gemini LLM integration** — natural language chat with CESA before and during the game
- **Stockfish engine** — adjustable skill level (0–20) for all abilities
- **Capture handling** — captured pieces are physically removed and placed in a graveyard row
- **Live robot control** — issue commands like "slow down" or "increase speed" mid-game via chat
- **Pre-game conversation** — chat with CESA freely; the game only starts when you say so
- **Voice support** — Whisper-based voice interaction (via `cesa_voice_chat.py`)

---

## Project Structure

```
CESA-Chess/
├── main.py                        ← Single entry point — run this
├── llm_robot_test.py              ← ABB YuMi robot arm + Gemini LLM control
├── cesa_voice_chat.py             ← Whisper voice chat integration
└── chessboard_codes/
    ├── hybrid_chess_system.py     ← Chess GUI + camera vision + Stockfish
    ├── chess_tracker.py           ← YOLO-based piece detection
    ├── chess_camera_bridge.py     ← Camera feed bridge (Raspberry Pi)
    ├── stockfish_engine.py        ← Stockfish wrapper
    ├── piece.py                   ← Chess piece definitions
    ├── utils.py                   ← Utility functions
    ├── bestV13.pt                 ← YOLO model weights
    └── res/                       ← GUI assets
```

---

## Hardware Requirements

| Component | Details |
|---|---|
| Robot Arm | ABB YuMi (IRB 14000), left arm |
| Camera | Raspberry Pi camera module via HTTP stream |
| Chess Board | Standard physical chessboard |
| Host PC | Windows 10/11, Python 3.11 |

---

## Software Requirements

- Python 3.11+
- [Stockfish](https://stockfishchess.org/download/) chess engine
- ABB RobotStudio / RAPID runtime

Install Python dependencies:

```bash
pip install chess pygame torch ultralytics google-generativeai openai-whisper requests numpy opencv-python
```

---

## Configuration

All configuration is at the top of `main.py`:

```python
PI_IP          = "10.223.92.247"          # Raspberry Pi IP (camera stream)
STREAM_URL     = f"http://{PI_IP}:5000/video_feed"
STOCKFISH_PATH = "C:\\Program Files\\stockfish\\..."
GEMINI_API_KEY = "your-api-key-here"
GEMINI_MODEL   = "gemini-2.5-flash"
```

In `llm_robot_test.py`, configure your robot connection and define `PRESET_POSITIONS` for all chess squares, hover positions, and the capture graveyard area.

---

## Running CESA

```bash
python main.py
```

You will be prompted for:
- **Skill level** (0–20, where 20 is Stockfish at full strength)
- **Enable robot arm** (y/n — run without robot for chess-only mode)

CESA will then greet you and chat. Say something like:

> *"Let's play a game"* or *"I'm ready"* or *"Challenge me"*

...and CESA will launch the chess system and open the board window.

---

## Chat Commands During the Game

You can type to CESA at any time while the game is running:

| What you say | What happens |
|---|---|
| `"what should I play next?"` | CESA gives chess advice based on the current position |
| `"increase speed"` | Robot arm moves faster |
| `"slow down"` | Robot arm slows down |
| `"move to pick position"` | Robot moves to the pick preset |
| `"open the gripper"` | Gripper opens |
| `"emergency stop"` | Robot stops immediately |
| `"quit chat"` | Closes the chat (game continues) |

---

## How Captures Work

When CESA captures a piece:
1. The captured piece is picked up and moved to a graveyard row beside the board
2. Each subsequent capture is placed 28.09mm further along the X axis, forming a neat row
3. CESA's piece is then moved to the now-empty destination square

The graveyard resets at the start of each new game.

---

## Robot Move Sequence

Each piece movement follows a strict 12-step sequence enforced by grounded single-step prompts to Gemini, preventing the LLM from skipping or combining steps:

1. Open gripper (10mm)
2. Move to safe pick position
3. Open gripper (9mm)
4. Hover above source square
5. Move down to source square
6. Close gripper (grip piece)
7. Move back up to hover height
8. Hover above destination square
9. Move down to destination square
10. Open gripper (release piece)
11. Move back up to hover height
12. Return to prep position

---

## Architecture

```
main.py  (Orchestrator)
│
├── RobotWorker (thread)
│     └── LLMRobotChat  ──► Gemini ──► ABB YuMi RAPID
│
├── GeminiChatWorker (thread)
│     └── Gemini (separate client) ──► terminal chat + robot commands
│
├── CESAVoiceChat (thread)
│     └── Whisper ──► voice-to-text ──► Gemini
│
└── HybridChessSystem (main thread, pygame)
      ├── StockfishEngine  ──► move calculation
      └── ChessTracker     ──► YOLO camera vision
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Stockfish](https://stockfishchess.org/) — open-source chess engine
- [Google Gemini](https://deepmind.google/technologies/gemini/) — LLM powering CESA's reasoning and robot control
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — computer vision for piece detection
- [ABB Robotics](https://new.abb.com/products/robotics) — YuMi robot arm
- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
