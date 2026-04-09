"""
main.py  –  CESA Chess + YuMi Robot  –  Single Entry Point
===========================================================

Project layout (run main.py from the root folder):

    main.py                          ← YOU RUN THIS
    llm_robot_test.py                ← robot arm + Gemini LLM
    chessboard_codes/
        hybrid_chess_system.py       ← chess GUI + vision + Stockfish
        chess_tracker.py
        chess_camera_bridge.py
        stockfish_engine.py
        piece.py
        utils.py
        res/

What it does
------------
1. Starts the ABB YuMi robot arm (llm_robot_test.py)
2. Starts the CESA chess system with camera vision (chessboard_codes/hybrid_chess_system.py)
3. Every time CESA (Stockfish / White) picks a move, this script pulls that
   move out and tells the Gemini LLM to physically execute it on the robot.
4. The pygame GUI and camera vision continue to work exactly as before.

Architecture
------------
HybridChessSystem  ──► (move detected)
        │
        ▼
  MoveWatcher thread  ──► LLMRobotChat.send_message("move e2 to e4")
        │
        ▼
   Gemini + RobotController  ──►  YuMi arm moves the piece
"""

import sys
import os
import time
import threading
import chess

# ── Root folder (where main.py lives) ───────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
CHESS_DIR      = os.path.join(ROOT_DIR, "chessboard_codes")

# Make both folders importable
sys.path.insert(0, ROOT_DIR)        # for llm_robot_test.py
sys.path.insert(0, CHESS_DIR)       # for hybrid_chess_system.py and its siblings


# ── Root folder (where main.py lives) ───────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
CHESS_DIR = os.path.join(ROOT_DIR, "chessboard_codes")

# Make both folders importable
sys.path.insert(0, ROOT_DIR)    # for llm_robot_test.py
sys.path.insert(0, CHESS_DIR)   # for hybrid_chess_system.py and its siblings



# ────────────────────────────────────────────────────────────────────────────
# 1.  Lazy-patch HybridChessSystem so we can intercept moves
#     without touching hybrid_chess_system.py at all.
# ────────────────────────────────────────────────────────────────────────────

def _patch_chess_system(chess_system, on_cesa_move_callback):
    """
    Monkey-patch make_stockfish_move() so that after CESA picks a move and
    the GUI is updated, our callback is invoked with (from_sq, to_sq, captured_sq).

    Capture detection is done BEFORE the move is applied (board still has the
    captured piece on it), so we know exactly which square to clear first.

    This requires zero changes to hybrid_chess_system.py.
    """
    original_make_stockfish_move = chess_system.make_stockfish_move

    def patched_make_stockfish_move():
        # ── Snapshot board state BEFORE the move ────────────────────────
        board_snapshot = chess_system.stockfish.board.copy()
        moves_before   = len(list(chess_system.stockfish.board.move_stack))

        result = original_make_stockfish_move()

        # ── Extract the move that was just played ────────────────────────
        moves_after = len(list(chess_system.stockfish.board.move_stack))
        if result and moves_after > moves_before:
            last_move = chess_system.stockfish.board.peek()
            from_sq   = chess.square_name(last_move.from_square)
            to_sq     = chess.square_name(last_move.to_square)

            # ── Detect capture using the PRE-move snapshot ───────────────
            captured_sq = None

            target_piece = board_snapshot.piece_at(last_move.to_square)
            if target_piece is not None:
                # Normal capture – piece was sitting on the destination square
                captured_sq = to_sq
                print(f"[Orchestrator] ⚔️  Capture detected: {target_piece.symbol()} on {captured_sq}")

            elif (board_snapshot.ep_square is not None
                  and last_move.to_square == board_snapshot.ep_square):
                # En-passant – captured pawn is one rank behind the destination
                moving_piece = board_snapshot.piece_at(last_move.from_square)
                if moving_piece and moving_piece.piece_type == chess.PAWN:
                    direction  = -1 if moving_piece.color == chess.WHITE else 1
                    ep_rank    = chess.square_rank(last_move.to_square) + direction
                    ep_file    = chess.square_file(last_move.to_square)
                    captured_sq = chess.square_name(chess.square(ep_file, ep_rank))
                    print(f"[Orchestrator] ⚔️  En-passant capture: pawn on {captured_sq}")

            try:
                on_cesa_move_callback(from_sq, to_sq, captured_sq)
            except Exception as e:
                print(f"[Orchestrator] ⚠️  Robot callback error: {e}")

        return result

    chess_system.make_stockfish_move = patched_make_stockfish_move
    print("[Orchestrator] ✓ Chess system patched – robot callbacks active")


# ────────────────────────────────────────────────────────────────────────────
# 2.  Robot worker  –  runs in its own thread so it never blocks the GUI
# ────────────────────────────────────────────────────────────────────────────

class RobotWorker:
    """
    Owns the RobotController + LLMRobotChat instances.
    Accepts move requests from the main thread and executes them
    sequentially in a background thread.
    """

    def __init__(self, skill_level: int):
        self.skill_level  = skill_level
        self._llm_chat    = None          # created in _init_thread
        self._ready       = threading.Event()
        self._failed      = False
        self._move_queue  = []
        self._queue_lock  = threading.Lock()
        self._queue_event = threading.Event()

        # Start initialisation + execution on a dedicated thread
        self._thread = threading.Thread(
            target=self._run, name="RobotWorker", daemon=True
        )
        self._thread.start()

    # ── public API (called from main thread) ────────────────────────────

    def wait_until_ready(self, timeout=30.0) -> bool:
        """Block until the robot arm is initialised (or timeout)."""
        return self._ready.wait(timeout)

    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._failed

    def queue_move(self, from_sq: str, to_sq: str, captured_sq: str = None):
        """
        Queue a chess move for the robot to execute.
        Returns immediately; execution happens in the background.
        """
        with self._queue_lock:
            self._move_queue.append((from_sq, to_sq, captured_sq))
        self._queue_event.set()
        capture_info = f" (captures {captured_sq})" if captured_sq else ""
        print(f"[RobotWorker] ✉  Move queued: {from_sq} → {to_sq}{capture_info}")

    # ── internal ────────────────────────────────────────────────────────

    def _run(self):
        """
        Queue-drain loop.  When constructed normally via __init__, first calls
        _init_robot().  When constructed manually by the launcher (via __new__)
        and _ready is already set, skips straight to the drain loop.
        """
        # If the launcher pre-built this worker, _ready is already set.
        # Only call _init_robot if we haven't been initialised yet.
        if not self._ready.is_set():
            if not self._init_robot():
                self._failed = True
                self._ready.set()
                return
            self._ready.set()
            print("[RobotWorker] ✅ Ready – waiting for moves…\n")

        # ── drain loop ───────────────────────────────────────────────────
        while True:
            self._queue_event.wait()
            self._queue_event.clear()

            while True:
                with self._queue_lock:
                    if not self._move_queue:
                        break
                    from_sq, to_sq, captured_sq = self._move_queue.pop(0)

                self._execute_move(from_sq, to_sq, captured_sq)

    def _init_robot(self) -> bool:
        """Import llm_robot_test, start RAPID, create LLMRobotChat."""
        try:
            from llm_robot_test import (
                start_rapid,
                set_speed,
                add_all_chess_squares_to_presets,
                LLMRobotChat,
            )

            print("[RobotWorker] Generating chess square positions…")
            add_all_chess_squares_to_presets()

            print("[RobotWorker] Starting RAPID…")
            if not start_rapid():
                print("[RobotWorker] ❌ Failed to start RAPID")
                return False

            time.sleep(1.5)
            set_speed(95)
            print("[RobotWorker] Speed set to 30 %")

            print("[RobotWorker] Initialising Gemini + RobotController…")
            self._llm_chat = LLMRobotChat()
            return True

        except Exception as e:
            print(f"[RobotWorker] ❌ Init failed: {e}")
            return False

    # Move counter – used to trigger periodic history resets
    _move_counter = 0

    # Capture counter – increments each time a piece is captured
    _capture_index = 0

    def _get_capture_position(self) -> tuple:
        """
        Calculate the hover and down preset names for the next captured piece.
        Each capture is offset -28.09mm in X from the base 'capture' position
        so pieces form a neat row rather than stacking on top of each other.

        Pattern:
          slot 0  →  X = 493.34  (base capture_down X)
          slot 1  →  X = 465.25  (493.34 - 28.09)
          slot 2  →  X = 437.16  (465.25 - 28.09)
          ...and so on

        Dynamically registers 'capture_slot_N' and 'capture_slot_N_down' into
        PRESET_POSITIONS in llm_robot_test.py so move_to_preset() can find them.

        Returns (hover_name, down_name) e.g. ("capture_slot_2", "capture_slot_2_down")
        """
        from llm_robot_test import PRESET_POSITIONS

        base      = PRESET_POSITIONS["capture"]
        base_down = PRESET_POSITIONS["capture_down"]
        idx       = RobotWorker._capture_index
        offset_x  = idx * -28.09   # -28.09mm per piece along X axis

        hover_name = f"capture_slot_{idx}"
        down_name  = f"capture_slot_{idx}_down"

        # Register hover position dynamically
        PRESET_POSITIONS[hover_name] = {
            "x":           base["x"] + offset_x,
            "y":           base["y"],
            "z":           base["z"],
            "q1":          base["q1"],
            "q2":          base["q2"],
            "q3":          base["q3"],
            "q4":          base["q4"],
            "description": f"Captured piece slot {idx} (hover, X={base['x'] + offset_x:.2f}mm)",
        }

        # Register down position dynamically
        PRESET_POSITIONS[down_name] = {
            "x":           base_down["x"] + offset_x,
            "y":           base_down["y"],
            "z":           base_down["z"],
            "q1":          base_down["q1"],
            "q2":          base_down["q2"],
            "q3":          base_down["q3"],
            "q4":          base_down["q4"],
            "description": f"Captured piece slot {idx} (down, X={base_down['x'] + offset_x:.2f}mm)",
        }

        print(f"[RobotWorker]   📍 Capture slot {idx}: "
              f"X={base_down['x'] + offset_x:.2f}mm  ({down_name})")

        RobotWorker._capture_index += 1
        return hover_name, down_name

    def _execute_move(self, from_sq: str, to_sq: str, captured_sq: str = None):
        """
        Execute a chess move by sending one grounded single-step prompt to
        Gemini per robot command.  This prevents Gemini from skipping steps
        because each message tells it exactly ONE thing to do next, leaving
        no room to combine or shortcut steps.

        Also resets the LLMRobotChat conversation history every 10 moves so
        that context drift (which causes step-skipping after ~15 moves) never
        accumulates.
        """
        print(f"\n[RobotWorker] 🦾 Executing move: {from_sq} → {to_sq}"
              + (f"  (capture on {captured_sq})" if captured_sq else ""))

        # ── Periodic history reset to prevent context drift ───────────────
        RobotWorker._move_counter += 1
        if RobotWorker._move_counter % 10 == 0:
            self._reset_llm_history()

        try:
            if captured_sq:
                # ── Phase 1: remove captured piece to its unique slot ─────
                hover_name, down_name = self._get_capture_position()
                print(f"[RobotWorker]   ⚔️  Phase 1: clearing {captured_sq} → {hover_name}")
                self._execute_chess_move_steps(captured_sq, hover_name, phase=1)
                print(f"[RobotWorker]   ✅ Phase 1 complete")

            # ── Phase 2: move CESA's piece (12 grounded steps) ───────────
            print(f"[RobotWorker]   ♟  Phase 2: {from_sq} → {to_sq}")
            self._execute_chess_move_steps(from_sq, to_sq, phase=2)
            print(f"[RobotWorker] ✅ Move complete: {from_sq} → {to_sq}\n")

        except Exception as e:
            print(f"[RobotWorker] ❌ Move failed ({from_sq}→{to_sq}): {e}")

    def _execute_chess_move_steps(self, from_sq: str, to_sq: str, phase: int):
        """
        Send one grounded prompt per step of the 12-step chess_move sequence.
        Each prompt tells Gemini exactly one robot command to execute —
        no multi-step instructions, no room to skip or combine steps.
        """

        steps = [
            (
                f"Step 1/12: Open the gripper to 10mm to prepare for picking.",
                f'robot.move_gripper(10)'
            ),
            (
                f"Step 2/12: Move to the safe pick position.",
                f'robot.move_to_preset("pick")'
            ),
            (
                f"Step 3/12: Open the gripper to 9mm ready for approach.",
                f'robot.move_gripper(9)'
            ),
            (
                f"Step 4/12: Move to hover above square {from_sq} (hover height, do NOT go down yet).",
                f'robot.move_to_preset("{from_sq}")'
            ),
            (
                f"Step 5/12: Move straight down to {from_sq}_down to reach the piece.",
                f'robot.move_to_preset("{from_sq}_down")'
            ),
            (
                f"Step 6/12: Close the gripper to grip the piece on {from_sq}.",
                f'robot.close_gripper()'
            ),
            (
                f"Step 7/12: Move straight back up to {from_sq} hover height (piece is gripped).",
                f'robot.move_to_preset("{from_sq}")'
            ),
            (
                f"Step 8/12: Move horizontally to hover above square {to_sq} (do NOT go down yet).",
                f'robot.move_to_preset("{to_sq}")'
            ),
            (
                f"Step 9/12: Move straight down to {to_sq}_down to place the piece.",
                f'robot.move_to_preset("{to_sq}_down")'
            ),
            (
                f"Step 10/12: Open the gripper to 9mm to release the piece on {to_sq}.",
                f'robot.open_gripper(9)'
            ),
            (
                f"Step 11/12: Move straight back up to {to_sq} hover height.",
                f'robot.move_to_preset("{to_sq}")'
            ),
            (
                f"Step 12/12: Return to the prep_grip position. This completes the move.",
                f'robot.move_to_preset("prep_grip")'
            ),
        ]

        total = len(steps)
        for i, (instruction, command) in enumerate(steps, 1):
            grounded_prompt = (
                f"[CHESS MOVE — Phase {phase}, {i}/{total}]\n"
                f"{instruction}\n\n"
                f"Execute ONLY this single robot command now, nothing else:\n"
                f"{command}"
            )
            print(f"[RobotWorker]     → {i}/{total}: {command}")
            self._llm_chat.send_message(grounded_prompt)

    def _reset_llm_history(self):
        """
        Clear the LLMRobotChat conversation history to prevent context drift.
        """
        print("[RobotWorker] 🔄 Resetting LLM conversation history to prevent drift…")
        if hasattr(self._llm_chat, 'history'):
            self._llm_chat.history = []
        if hasattr(self._llm_chat, 'conversation_history'):
            self._llm_chat.conversation_history = []
        print("[RobotWorker] ✅ History reset complete")

    def reset_game(self):
        """
        Reset all game-scoped counters when a new game starts.
        """
        RobotWorker._capture_index = 0
        RobotWorker._move_counter  = 0
        self._reset_llm_history()
        print("[RobotWorker] 🔄 Game reset: capture slots cleared")


# ────────────────────────────────────────────────────────────────────────────
# 3.  Gemini Chat Worker  –  live terminal chat, runs on its own thread
# ────────────────────────────────────────────────────────────────────────────

class GeminiChatWorker:
    """
    Runs an interactive Gemini chat in the terminal on a background thread.
    Stays aware of live game state AND can execute robot commands directly.

    If CESA's reply contains lines starting with 'robot.' they are extracted
    and forwarded to RobotWorker._llm_chat.execute_command() so commands like
    "increase speed", "slow down", "go to prep_grip" work instantly mid-game.
    """

    SYSTEM_PROMPT = """\
You are CESA - Cognitive Embodied Strategic Agent - an AI chess assistant
embedded in a live chess system. A human is playing chess against a
Stockfish engine (also called CESA) which controls an ABB YuMi robot arm
that physically moves pieces on a real board.

You have access to the current game state and can answer any question the
human asks - chess strategy, robot operation, general knowledge, anything.
Be concise, friendly, and helpful. When discussing the current game, refer
to the live state provided at the start of each message.

ROBOT CONTROL:
You can also control the robot directly during the game. If the human asks
you to do something physical (change speed, move to a position, open/close
gripper, etc.), respond conversationally AND include the robot command on
its own line starting with "robot."

Available robot commands during play:
  robot.set_speed(percent)          - e.g. robot.set_speed(50)
  robot.move_to_preset("name")      - e.g. robot.move_to_preset("pick")
  robot.open_gripper(width)         - e.g. robot.open_gripper(10)
  robot.close_gripper()
  robot.move_gripper(position)      - e.g. robot.move_gripper(8)
  robot.emergency_stop()            - ONLY if human explicitly requests it

RULES:
- Only include robot commands when the human is asking for robot control.
- For chess questions or general chat, reply conversationally with no robot commands.
- Never include robot.chess_move() - chess moves are handled automatically.
- One robot command per line, no markdown backticks around them.
"""

    def __init__(self, game_state_fn, api_key: str, model: str,
                 robot_worker=None):
        self._game_state_fn  = game_state_fn
        self._api_key        = api_key
        self._model          = model
        self._robot_worker   = robot_worker
        self._history        = []
        self._stop_event     = threading.Event()
        self._thread         = threading.Thread(
            target=self._run, name="GeminiChat", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    # ── internal ────────────────────────────────────────────────────────

    def _execute_robot_command(self, command: str):
        if self._robot_worker is None or self._robot_worker._llm_chat is None:
            print(f"[GeminiChat] Robot not available, cannot execute: {command}")
            return
        print(f"[GeminiChat] Executing robot command: {command}")
        try:
            self._robot_worker._llm_chat.execute_command(command)
        except Exception as e:
            print(f"[GeminiChat] Robot command failed: {e}")

    def _run(self):
        """Initialise Gemini then loop on terminal input."""
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=self._api_key)
            client.models.generate_content(model=self._model, contents="OK")
        except Exception as e:
            print(f"\n[GeminiChat] Failed to connect: {e}")
            return

        print("\n" + "-" * 60)
        print("GEMINI CHAT is ready!  Type your question below.")
        print("   The game continues running while you chat.")
        print("   Robot commands (speed, position, etc.) are executed live.")
        print("   Type 'quit chat' to close this chat.")
        print("-" * 60 + "\n")

        while not self._stop_event.is_set():
            try:
                user_input = input("YOU > ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() == "quit chat":
                print("[GeminiChat] Chat closed.")
                break

            game_state   = self._game_state_fn()
            full_message = (
                f"[LIVE GAME STATE]\n{game_state}\n\n"
                f"[USER QUESTION]\n{user_input}"
            )

            self._history.append({"role": "user", "content": full_message})
            if len(self._history) > 20:
                self._history = self._history[-20:]

            try:
                conversation = self.SYSTEM_PROMPT + "\n\n"
                for msg in self._history:
                    prefix = "Human: " if msg["role"] == "user" else "CESA: "
                    conversation += prefix + msg["content"] + "\n\n"

                response = client.models.generate_content(
                    model=self._model,
                    contents=conversation,
                )
                reply = response.text.strip() if response and response.text else "(no response)"
                reply = reply.replace("```python", "").replace("```", "").strip()

                self._history.append({"role": "assistant", "content": reply})

                # Split conversational text from robot commands
                chat_lines = []
                robot_cmds = []
                for line in reply.splitlines():
                    if line.strip().startswith("robot."):
                        robot_cmds.append(line.strip())
                    elif line.strip():
                        chat_lines.append(line.strip())

                if chat_lines:
                    print(f"\nCESA > {' '.join(chat_lines)}\n")

                if robot_cmds:
                    print(f"[GeminiChat] Executing {len(robot_cmds)} robot command(s)...")
                    for cmd in robot_cmds:
                        self._execute_robot_command(cmd)

            except Exception as e:
                print(f"\n[GeminiChat] Error: {e}\n")


# ────────────────────────────────────────────────────────────────────────────
# 4.  Orchestrator  –  ties everything together
# ────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Single entry point.  Creates both sub-systems and glues them together.
    """

    def __init__(
        self,
        stream_url:      str,
        stockfish_path:  str  = None,
        skill_level:     int  = 5,
        robot_enabled:   bool = True,
        gemini_api_key:  str  = "",
        gemini_model:    str  = "gemini-2.5-flash",
    ):
        self.stream_url      = stream_url
        self.stockfish_path  = stockfish_path
        self.skill_level     = skill_level
        self.robot_enabled   = robot_enabled
        self.gemini_api_key  = gemini_api_key
        self.gemini_model    = gemini_model

        self._robot_worker   = None
        self._chess_system   = None
        self._chat_worker    = None

        # Track move history for the chat worker to read
        self._move_history   = []
        self._history_lock   = threading.Lock()

    # ── game state snapshot (called by GeminiChatWorker before each reply) ──

    def _get_game_state(self) -> str:
        """Return a human-readable snapshot of the current game for Gemini."""
        lines = []
        if self._chess_system is None:
            return "Game not started yet."

        try:
            board  = self._chess_system.stockfish.board
            turn   = "Black (Human)" if board.turn == chess.BLACK else "White (CESA)"
            move_n = self._chess_system.move_count // 2

            lines.append(f"Move number  : {move_n}")
            lines.append(f"Turn         : {turn}")
            lines.append(f"In check     : {board.is_check()}")
            lines.append(f"Game over    : {self._chess_system.game_over}")
            lines.append(f"FEN          : {board.fen()}")

            with self._history_lock:
                recent = self._move_history[-10:]
            if recent:
                lines.append(f"Recent moves : {', '.join(recent)}")
            else:
                lines.append("Recent moves : (none yet)")

        except Exception as e:
            lines.append(f"(error reading game state: {e})")

        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────

    def run(self):
        """Start everything.  Blocks until the chess GUI is closed."""

        # ── Step 1: start robot worker in background ─────────────────────
        if self.robot_enabled:
            if self._robot_worker is not None:
                # ── Robot already injected by launcher — skip init ────────
                print("=" * 60)
                print("STEP 1/3 – Robot already initialised (reusing existing).")
                print("=" * 60)
            else:
                # ── Normal standalone path — start RAPID fresh ────────────
                print("=" * 60)
                print("STEP 1/3 – Starting YuMi robot arm (background)…")
                print("=" * 60)
                self._robot_worker = RobotWorker(self.skill_level)

                print("Waiting for robot arm to initialise (up to 30 s)…")
                ready = self._robot_worker.wait_until_ready(timeout=30.0)
                if not ready or not self._robot_worker.is_ready():
                    print("⚠️  Robot arm failed to initialise – continuing WITHOUT robot.")
                    self._robot_worker = None
        else:
            print("ℹ️  Robot disabled – chess-only mode.")

        # ── Step 2: start Gemini chat in background ───────────────────────
        print("\n" + "=" * 60)
        print("STEP 2/3 – Starting Gemini chat worker…")
        print("=" * 60)
        self._chat_worker = GeminiChatWorker(
            game_state_fn = self._get_game_state,
            api_key       = self.gemini_api_key,
            model         = self.gemini_model,
            robot_worker  = self._robot_worker,
        )
        self._chat_worker.start()

        # ── Step 3: create chess system ───────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 3/3 – Starting CESA chess system…")
        print("=" * 60)

        os.chdir(CHESS_DIR)
        print(f"[Orchestrator] Working directory set to: {CHESS_DIR}")

        from hybrid_chess_system import HybridChessSystem

        self._chess_system = HybridChessSystem(
            stream_url     = self.stream_url,
            stockfish_path = self.stockfish_path,
            skill_level    = self.skill_level,
        )

        # ── Step 4: patch chess system to fire robot callbacks ─────────────
        if self._robot_worker is not None:
            _patch_chess_system(self._chess_system, self._on_cesa_move)
            _original_reset = self._chess_system.reset_game
            _worker         = self._robot_worker
            _orchestrator   = self
            def _patched_reset():
                _original_reset()
                _worker.reset_game()
                with _orchestrator._history_lock:
                    _orchestrator._move_history.clear()
                print("[Orchestrator] 🔄 New game: capture slots and move history cleared")
            self._chess_system.reset_game = _patched_reset
        else:
            print("[Orchestrator] Robot not available – skipping patch.")

        # ── Step 5: wait for user confirmation before launching the GUI ────
        print("\n" + "=" * 60)
        print("✅ All systems initialised and ready.")
        print("─" * 60)
        print("💬 Chat with CESA in this terminal while the game runs!")
        print("─" * 60)
        while True:
            confirm = input("\nType 'yes' or 'chess' to start the game: ").strip().lower()
            if confirm in ("yes", "chess"):
                break
            print("  (Waiting… type 'yes' or 'chess' to begin)")

        # ── Step 6: hand control to the chess GUI (blocks here) ────────────
        print("\n" + "=" * 60)
        print("Starting game – close the window to quit.")
        print("=" * 60 + "\n")
        self._chess_system.run()

        self._chat_worker.stop()
        print("\n[Orchestrator] Chess system exited.  Goodbye!")

    # ── callbacks ────────────────────────────────────────────────────────

    def _on_cesa_move(self, from_sq: str, to_sq: str, captured_sq: str = None):
        """
        Called every time CESA picks a move.
        Queues the move for the robot and logs it for the chat worker.
        Returns instantly so the GUI is never blocked.
        """
        capture_info = f" (captures {captured_sq})" if captured_sq else ""
        print(f"\n[Orchestrator] 🔔 CESA move detected: {from_sq} → {to_sq}{capture_info}")

        with self._history_lock:
            label = f"CESA:{from_sq}→{to_sq}" + (f"x{captured_sq}" if captured_sq else "")
            self._move_history.append(label)

        if self._robot_worker is not None:
            self._robot_worker.queue_move(from_sq, to_sq, captured_sq)


# ────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CESA Chess + YuMi Robot  –  Orchestrator")
    print("=" * 60)
    print()
    print("Files being used:")
    print(f"  Chess system : chessboard_codes/hybrid_chess_system.py")
    print(f"  Robot / LLM  : llm_robot_test.py")
    print()

    PI_IP        = "10.223.93.212"
    STREAM_URL   = f"http://{PI_IP}:5000/video_feed"
    STOCKFISH_PATH = (
        "C:\\Program Files\\stockfish\\stockfish\\"
        "stockfish-windows-x86-64-avx2.exe"
    )
    GEMINI_API_KEY = "AIzaSyDBraZf7UgDFg8YxP0GCtht83GvnKthqnQ"
    GEMINI_MODEL   = "gemini-2.5-flash"

    raw = input("Enter CESA skill level (0-20, default 5): ").strip()
    skill_level = int(raw) if raw.isdigit() else 5
    skill_level = max(0, min(20, skill_level))

    raw = input("Enable YuMi robot arm? (y/n, default y): ").strip().lower()
    robot_enabled = (raw != "n")

    print(f"\nSettings:")
    print(f"  Stream URL    : {STREAM_URL}")
    print(f"  Skill level   : {skill_level}")
    print(f"  Robot enabled : {robot_enabled}")
    print(f"  Gemini chat   : enabled (text terminal)")
    print()

    orchestrator = Orchestrator(
        stream_url      = STREAM_URL,
        stockfish_path  = STOCKFISH_PATH,
        skill_level     = skill_level,
        robot_enabled   = robot_enabled,
        gemini_api_key  = GEMINI_API_KEY,
        gemini_model    = GEMINI_MODEL,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()