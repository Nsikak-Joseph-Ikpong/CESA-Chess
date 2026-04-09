"""
launcher.py  –  CESA Unified Launcher
======================================

Run this file to start CESA with voice control + robot arm first.
When you're ready to play chess, just say it — CESA will confirm and launch
the chess + robot system while keeping voice chat live throughout.

    python launcher.py

Flow
----
1. Initialises the robot arm immediately (so voice robot control works from the start)
2. Starts CESA voice chat — you can talk and control the robot by voice right away
3. Say "let's play chess", "start a game", "I want to play", etc.
4. CESA asks for verbal confirmation, then calls start_chess_game()
5. Chess + robot run in background — the SAME robot, no re-initialisation
6. Say "stop the game" / "end the chess game" to stop chess cleanly

Voice commands always available (robot arm)
--------------------------------------------
  All robot commands from cesa_voice_chat.py work from the moment CESA starts.
  "move to e4", "go to pick", "grab the piece", "slow down", etc.

Chess control (voice)
----------------------
  "let's play chess"    → CESA confirms, then starts
  "start a game"        → same
  "stop the game"       → ends chess session cleanly
"""

import sys
import os
import time
import threading
import signal
import json
import importlib.util

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
CHESS_DIR = os.path.join(ROOT_DIR, "chessboard_codes")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHESS_DIR)

from google.genai import types
from cesa_voice_chat import CESAVoiceChat, _SYSTEM_PROMPT, _ROBOT_TOOLS

# ── Config ────────────────────────────────────────────────────────────────────
PI_IP          = "10.223.93.212"
STREAM_URL     = f"http://{PI_IP}:5000/video_feed"
GEMINI_API_KEY = "AIzaSyDBraZf7UgDFg8YxP0GCtht83GvnKthqnQ"
GEMINI_MODEL   = "gemini-2.5-flash"
STOCKFISH_PATH = (
    "C:\\Program Files\\stockfish\\stockfish\\"
    "stockfish-windows-x86-64-avx2.exe"
)

# ── Extended system prompt ────────────────────────────────────────────────────
_LAUNCHER_SYSTEM_PROMPT = _SYSTEM_PROMPT + """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHESS GAME LAUNCHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You can start and stop a physical chess game on the robot board.

STARTING A GAME:
- If the human says anything suggesting they want to play chess
  ("let's play", "fancy a game", "start chess", "I want to play you",
   "set up the board", "let's have a match", etc.), call confirm_chess_start().
- confirm_chess_start() prompts you to ask the human to confirm verbally.
- Only call start_chess_game() AFTER the human explicitly confirms
  (says "yes", "go ahead", "sure", "let's do it", etc.).
- Do NOT call start_chess_game() without first calling confirm_chess_start()
  and receiving clear verbal confirmation.
- Once started, say: "The board is initialising — I'll let you know when it's
  ready." Then stop. Don't describe the process in detail.

STOPPING A GAME:
- If the human says "stop the game", "end the game", "quit chess", or similar
  → call stop_chess_game().
- Confirm in one sentence: "Stopping the game now."

DURING A GAME:
- You can still chat about anything while chess runs in the background.
- You still control the robot arm with all the usual voice commands.
- The chess system handles piece detection and moves automatically.
"""

# ── Chess control tool declarations ──────────────────────────────────────────
_CHESS_TOOLS = [
    types.Tool(function_declarations=[

        types.FunctionDeclaration(
            name="confirm_chess_start",
            description=(
                "Call this when the human expresses any intent to play chess. "
                "Signals CESA to ask for verbal confirmation before launching. "
                "Do NOT call start_chess_game until the human has said yes."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
                required=[]
            )
        ),

        types.FunctionDeclaration(
            name="start_chess_game",
            description=(
                "Launch the full CESA chess + robot system. "
                "Only call this AFTER the human has explicitly confirmed "
                "(said yes, go ahead, sure, etc.). "
                "Optionally specify skill level (0-20, default 5)."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "skill_level": types.Schema(
                        type=types.Type.NUMBER,
                        description="Stockfish skill level 0-20 (default 5)"
                    ),
                },
                required=[]
            )
        ),

        types.FunctionDeclaration(
            name="stop_chess_game",
            description=(
                "Stop the running chess game and shut down cleanly. "
                "Call this when the human asks to stop, end, or quit chess."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
                required=[]
            )
        ),

    ])
]


# ── CESALauncher ──────────────────────────────────────────────────────────────

class CESALauncher(CESAVoiceChat):
    """
    Extends CESAVoiceChat by:
      1. Overriding the system prompt and tools to include chess control
      2. Overriding _dispatch_tool to handle chess tools, passing everything
         else up to the parent (robot arm tools)
      3. Managing the Orchestrator lifecycle in a background thread

    KEY FIX: The robot arm is initialised ONCE in main() and passed in.
    When chess launches, the same robot is injected directly into the
    Orchestrator's RobotWorker — RAPID is never started a second time.
    """

    def __init__(self, api_key: str, stream_url: str, robot=None,
                 skill_level: int = 5):
        super().__init__(
            api_key    = api_key,
            stream_url = stream_url,
            robot      = robot,
        )
        self._default_skill_level   = skill_level
        self._orchestrator          = None
        self._orchestrator_thread   = None
        self._chess_running         = False
        self._awaiting_confirmation = False

    # ── Override _voice_session to inject extended prompt + combined tools ────

    async def _voice_session(self):
        """
        Same as parent _voice_session but uses the extended system prompt
        and combined tool list (robot tools + chess control tools).
        All session logic (send/receive/audio) is identical to the parent.
        """
        import asyncio
        import pyaudio
        import time as _time

        from google import genai
        from google.genai import types as _types
        from cesa_voice_chat import (
            FORMAT, CHANNELS, MIC_RATE, SPK_RATE, CHUNK_SIZE,
            POST_SPEECH_MUTE, FRAME_INTERVAL, MODEL_ID,
        )

        pya    = pyaudio.PyAudio()
        client = genai.Client(
            api_key=self._api_key,
            http_options={"api_version": "v1alpha"},
        )

        # Combined tool list: robot tools (if connected) + chess control tools
        all_decls = []
        if self._robot is not None:
            for tool_obj in _ROBOT_TOOLS:
                all_decls.extend(tool_obj.function_declarations)
        for tool_obj in _CHESS_TOOLS:
            all_decls.extend(tool_obj.function_declarations)

        combined_tools = [_types.Tool(function_declarations=all_decls)]

        config = _types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=_LAUNCHER_SYSTEM_PROMPT,
            tools=combined_tools,
            speech_config=_types.SpeechConfig(
                voice_config=_types.VoiceConfig(
                    prebuilt_voice_config=_types.PrebuiltVoiceConfig(
                        voice_name=self._voice_name
                    )
                )
            ),
        )

        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            with self._session_lock:
                self._session = session

            print("[CESALauncher] CESA is live — speak freely!\n")

            mic_stream = pya.open(
                format=FORMAT, channels=CHANNELS, rate=MIC_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE,
            )
            spk_stream = pya.open(
                format=FORMAT, channels=CHANNELS, rate=SPK_RATE,
                output=True,
            )

            audio_queue = asyncio.Queue()
            ai_speaking = asyncio.Event()
            mute_until  = 0.0

            async def send_audio():
                nonlocal mute_until
                while not self._stop_event.is_set():
                    data = await asyncio.to_thread(
                        mic_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    if _time.time() > mute_until:
                        await session.send_realtime_input(
                            media=_types.Blob(data=data, mime_type="audio/pcm"),
                        )
                    await asyncio.sleep(0)

            async def drain_move_queue():
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.1)
                    with self._move_lock:
                        pending = self._move_queue[:]
                        self._move_queue.clear()
                    for msg in pending:
                        try:
                            await session.send_client_content(
                                turns=_types.Content(
                                    role="user",
                                    parts=[_types.Part(text=msg)],
                                ),
                                turn_complete=True,
                            )
                            print("[CESALauncher] Move event sent to session")
                        except Exception as e:
                            print(f"[CESALauncher] Move event send failed: {e}")

            async def send_frames():
                while not self._stop_event.is_set():
                    await asyncio.sleep(FRAME_INTERVAL)
                    frame = self._get_latest_frame()
                    if frame:
                        try:
                            await session.send_realtime_input(
                                media=_types.Blob(data=frame, mime_type="image/jpeg"),
                            )
                        except Exception as e:
                            print(f"[CESALauncher] Frame send error: {e}")

            async def receive_and_handle():
                nonlocal mute_until
                processed_tool_ids = set()

                while not self._stop_event.is_set():
                    async for response in session.receive():

                        if response.tool_call:
                            for fn_call in response.tool_call.function_calls:
                                fn_name = fn_call.name
                                fn_args = dict(fn_call.args) if fn_call.args else {}
                                fn_id   = fn_call.id

                                if fn_id in processed_tool_ids:
                                    print(f"[CESALauncher] Skipping duplicate: {fn_name}")
                                    continue
                                processed_tool_ids.add(fn_id)

                                print(f"[CESALauncher] Tool call: {fn_name}({fn_args})")

                                result_str = await asyncio.to_thread(
                                    self._dispatch_tool, fn_name, fn_args
                                )

                                await session.send_tool_response(
                                    function_responses=[
                                        _types.FunctionResponse(
                                            id=fn_id,
                                            name=fn_name,
                                            response={"result": result_str},
                                        )
                                    ]
                                )
                                print(f"[CESALauncher] Tool result sent for '{fn_name}'")

                        if response.server_content:
                            if response.server_content.interrupted:
                                print("[CESALauncher] Interrupted by user")
                                ai_speaking.clear()
                                drained = 0
                                while not audio_queue.empty():
                                    try:
                                        audio_queue.get_nowait()
                                        drained += 1
                                    except asyncio.QueueEmpty:
                                        break
                                if drained:
                                    print(f"[CESALauncher] Drained {drained} audio chunks")

                            if response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if hasattr(part, "inline_data") and part.inline_data:
                                        ai_speaking.set()
                                        await audio_queue.put(part.inline_data.data)

                            if response.server_content.turn_complete:
                                ai_speaking.clear()
                                mute_until = _time.time() + POST_SPEECH_MUTE
                                print("[CESALauncher] CESA finished speaking")

            async def play_audio():
                while not self._stop_event.is_set():
                    data = await audio_queue.get()
                    await asyncio.to_thread(spk_stream.write, data)

            try:
                tasks = [
                    send_audio(),
                    receive_and_handle(),
                    play_audio(),
                    drain_move_queue(),
                ]
                if self._stream_url:
                    tasks.append(send_frames())

                await asyncio.gather(*tasks)

            finally:
                with self._session_lock:
                    self._session = None
                mic_stream.stop_stream()
                mic_stream.close()
                spk_stream.stop_stream()
                spk_stream.close()
                pya.terminate()

    # ── Tool dispatcher — chess tools here, robot tools to parent ─────────────

    def _dispatch_tool(self, name: str, args: dict) -> str:
        if name in ("confirm_chess_start", "start_chess_game", "stop_chess_game"):
            return self._dispatch_chess_tool(name, args)
        else:
            return super()._dispatch_tool(name, args)

    # ── Chess tool logic ──────────────────────────────────────────────────────

    def _dispatch_chess_tool(self, name: str, args: dict) -> str:

        if name == "confirm_chess_start":
            self._awaiting_confirmation = True
            print("[CESALauncher] Confirmation gate opened")
            return json.dumps({
                "success": True,
                "message": (
                    "Ask the human to confirm verbally. Say something like: "
                    "'Shall I set up the board? Just say yes and I'll get everything started.' "
                    "Wait for them to confirm before calling start_chess_game."
                )
            })

        elif name == "start_chess_game":
            if self._chess_running:
                return json.dumps({
                    "success": False,
                    "message": "Chess is already running."
                })
            skill_level = int(args.get("skill_level", self._default_skill_level))
            print(f"[CESALauncher] Launching chess — skill={skill_level}")
            self._launch_chess(skill_level)
            return json.dumps({
                "success": True,
                "message": (
                    f"Chess system is launching at skill level {skill_level}. "
                    "The board will be ready in a moment."
                )
            })

        elif name == "stop_chess_game":
            if not self._chess_running:
                return json.dumps({
                    "success": False,
                    "message": "No chess game is currently running."
                })
            self._stop_chess()
            return json.dumps({"success": True, "message": "Chess game stopped."})

        else:
            return json.dumps({"success": False, "message": f"Unknown chess tool: {name}"})

    # ── Chess launch / stop ───────────────────────────────────────────────────

    def _launch_chess(self, skill_level: int):
        """
        Load Orchestrator from main.py and run it in a background thread.

        CRITICAL: The robot is already initialised and running from main().
        We manually construct a RobotWorker that reuses the existing RAPID
        session via a fresh LLMRobotChat — start_rapid() is never called again.
        """
        _spec = importlib.util.spec_from_file_location(
            "root_main", os.path.join(ROOT_DIR, "main.py")
        )
        _root_main = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_root_main)
        Orchestrator = _root_main.Orchestrator
        RobotWorker  = _root_main.RobotWorker

        has_robot = self._robot is not None

        def _run():
            self._chess_running = True
            print("\n[CESALauncher] ♟  Orchestrator starting…\n")
            try:
                orc = Orchestrator(
                    stream_url     = self._stream_url or "",
                    stockfish_path = STOCKFISH_PATH,
                    skill_level    = skill_level,
                    robot_enabled  = has_robot,   # tells Orchestrator.run() to expect a robot
                    gemini_api_key = self._api_key,
                    gemini_model   = GEMINI_MODEL,
                )
                self._orchestrator = orc

                # ── Build RobotWorker WITHOUT calling start_rapid() ──────────
                if has_robot:
                    print("[CESALauncher] Building RobotWorker from existing RAPID session…")
                    worker = RobotWorker.__new__(RobotWorker)
                    worker.skill_level  = skill_level
                    worker._failed      = False
                    worker._move_queue  = []
                    worker._queue_lock  = threading.Lock()
                    worker._queue_event = threading.Event()
                    worker._ready       = threading.Event()
                    worker._llm_chat    = None

                    try:
                        from llm_robot_test import LLMRobotChat
                        worker._llm_chat = LLMRobotChat()
                        worker._ready.set()
                        print("[CESALauncher] ✅ RobotWorker ready (reused existing RAPID)")
                    except Exception as e:
                        print(f"[CESALauncher] ⚠️  LLMRobotChat init failed: {e}")
                        worker._failed = True
                        worker._ready.set()

                    # Start the move-queue drain loop
                    worker._thread = threading.Thread(
                        target=worker._run, name="RobotWorker", daemon=True
                    )
                    worker._thread.start()

                    # Inject into Orchestrator BEFORE orc.run() patches things
                    orc._robot_worker = worker

                # ── Patch chess move callback to also announce via voice ──────
                _orig  = orc._on_cesa_move
                _voice = self
                def _patched(from_sq, to_sq, captured_sq=None):
                    _orig(from_sq, to_sq, captured_sq)
                    _voice.notify_move(from_sq, to_sq, captured_sq)
                orc._on_cesa_move = _patched

                # Give voice chat access to live game state
                self._game_state_fn = orc._get_game_state

                orc.run()   # blocks until the chess GUI is closed

            except Exception as e:
                print(f"[CESALauncher] Orchestrator error: {e}")
            finally:
                self._chess_running = False
                self._orchestrator  = None
                print("[CESALauncher] Chess session ended.")

        self._orchestrator_thread = threading.Thread(
            target=_run, name="OrchestratorThread", daemon=True
        )
        self._orchestrator_thread.start()

    def _stop_chess(self):
        """Signal the chess GUI to close, which unblocks orc.run()."""
        if self._orchestrator is None:
            return
        print("[CESALauncher] Stopping chess system…")
        try:
            chess_sys = self._orchestrator._chess_system
            if chess_sys is not None:
                chess_sys.game_over = True
                try:
                    import pygame
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    print("[CESALauncher] QUIT event posted to pygame")
                except Exception:
                    pass
        except Exception as e:
            print(f"[CESALauncher] Stop chess error: {e}")
        self._chess_running = False


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CESA  —  Voice + Robot + Chess  (Unified Launcher)")
    print("=" * 60)
    print()
    print("Phase 1 : Robot arm initialises now")
    print("Phase 2 : CESA voice chat starts — control the robot by voice")
    print("Phase 3 : Say 'let's play chess' to launch the chess system")
    print()

    raw = input("Enter CESA skill level (0-20, default 5): ").strip()
    skill_level = int(raw) if raw.isdigit() else 5
    skill_level = max(0, min(20, skill_level))

    raw = input("Enable YuMi robot arm? (y/n, default y): ").strip().lower()
    robot_enabled = (raw != "n")

    # ── Initialise robot arm ONCE up front ───────────────────────────────────
    robot = None
    if robot_enabled:
        print()
        print("Initialising robot arm…")
        try:
            from llm_robot_test import (
                start_rapid,
                set_speed,
                add_all_chess_squares_to_presets,
                RobotController,
            )

            print("  Generating chess square positions…")
            add_all_chess_squares_to_presets()

            print("  Starting RAPID…")
            if not start_rapid():
                print("  ⚠️  Failed to start RAPID — continuing without robot.")
            else:
                time.sleep(1.5)
                set_speed(30)
                print("  Speed set to 30 %")
                robot = RobotController()
                print("  ✅ Robot arm ready.")

        except Exception as e:
            print(f"  ⚠️  Robot init failed ({e}) — voice-only mode.")
            robot = None
    else:
        print("Robot disabled — voice-only mode.")

    print()
    print(f"  Camera     : {STREAM_URL}")
    print(f"  Skill      : {skill_level}")
    print(f"  Robot      : {'connected' if robot else 'disabled'}")
    print()
    print("Starting CESA voice session…")
    print("Speak freely — robot commands work now, say 'let's play chess' when ready.")
    print("Press Ctrl+C to quit.")
    print()

    launcher = CESALauncher(
        api_key     = GEMINI_API_KEY,
        stream_url  = STREAM_URL,
        robot       = robot,
        skill_level = skill_level,
    )
    launcher.start()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def _shutdown(sig, frame):
        print("\n[CESA] Shutting down…")
        launcher.stop()
        if robot_enabled and robot is not None:
            try:
                from llm_robot_test import hold_rapid, SESSION as ROBOT_SESSION
                hold_rapid()
                ROBOT_SESSION.close()
                print("[CESA] RAPID stopped.")
            except Exception:
                pass
        print("[CESA] Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()