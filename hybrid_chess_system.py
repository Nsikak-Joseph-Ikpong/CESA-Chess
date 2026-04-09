"""
Hybrid Chess System with Stockfish - Stateless Vision Model
- Stockfish is the SINGLE SOURCE OF TRUTH for game state
- Vision model detects occupancy only (black/white/empty)
- Compares camera occupancy with Stockfish's expected state to find human move
- Full game state notifications: Check, Checkmate, Stalemate, Illegal moves
- Manual fallback if camera fails
- VIDEO STREAM MODE: Analyzes live video after SPACE press for fast detection
"""

import cv2
import pygame
import threading
import numpy as np
from chess_tracker import ChessTracker
from chess_camera_bridge import ChessCameraBridge
from stockfish_engine import StockfishEngine
from piece import Piece
from utils import Utils
import time
import os
import chess

# Replace with your Pi's IP
PI_IP = "10.223.92.247"
STREAM_URL = f"http://{PI_IP}:5000/video_feed"


class HybridChessSystem:
    """Chess system with Stockfish AI, stateless camera detection and manual fallback"""

    def __init__(self, stream_url=STREAM_URL, stockfish_path=None, skill_level=5):
        """Initialize hybrid chess system with Stockfish"""
        self.stream_url = stream_url
        self.stockfish_path = stockfish_path
        self.skill_level = skill_level

        # Initialize Stockfish
        print("🤖 Initializing CESA (Cognitive Embodied Strategic Agent)...")
        try:
            self.stockfish = StockfishEngine(
                stockfish_path=stockfish_path,
                skill_level=skill_level,
                time_limit=2.0
            )
            print(f"✓ CESA ready (Skill Level: {skill_level})")
        except Exception as e:
            print(f"❌ Failed to initialize CESA: {e}")
            print("\n💡 Make sure Stockfish is installed:")
            print("   • Ubuntu/Debian: sudo apt install stockfish")
            print("   • macOS: brew install stockfish")
            raise

        # Initialize camera tracker (stateless - just for occupancy detection)
        self.tracker = ChessTracker()

        # Mode control
        self.in_manual_mode = False
        self.manual_mode_temporary = False

        # Threading controls
        self.running = True
        self.camera_ready = False
        self.game_started = False
        self.pause_camera_thread = False

        # Camera capture
        self.cap = None
        self.current_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()

        # Move tracking
        self.move_count = 0
        self.move_history = []  # Store move history

        # AI state
        self.waiting_for_human = False

        # Game state notifications
        self.notification_message = None
        self.notification_color = (255, 255, 255)
        self.notification_timer = 0
        self.game_over = False

        # Pygame/GUI setup
        self.screen_width = 1280
        self.screen_height = 750
        self.camera_display_size = (640, 640)
        self.board_offset_x = 640
        self.board_offset_y = 50

        # Initialize Pygame
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        pygame.display.set_caption("Chess - CESA vs Human")

        # Set icon
        self.resources = "res"
        icon_src = os.path.join(self.resources, "chess_icon.png")
        if os.path.exists(icon_src):
            icon = pygame.image.load(icon_src)
            pygame.display.set_icon(icon)

        self.clock = pygame.time.Clock()

        # Load board image
        board_src = os.path.join(self.resources, "board.png")
        self.board_img = pygame.image.load(board_src).convert()

        # Calculate board locations
        square_length = self.board_img.get_rect().width // 8
        self.square_length = square_length

        self.board_locations = []
        for x in range(0, 8):
            self.board_locations.append([])
            for y in range(0, 8):
                self.board_locations[x].append([
                    self.board_offset_x + (x * square_length),
                    self.board_offset_y + (y * square_length)
                ])

        # Initialize chess game with camera bridge
        pieces_src = os.path.join(self.resources, "pieces.png")
        self.chess = ChessCameraBridge(
            self.screen, pieces_src, self.board_locations, square_length
        )

        # Utils for mouse handling
        self.utils = Utils()

        print("✓ Pygame initialized")

    def show_notification(self, message, color=(255, 255, 255), duration=3.0):
        """
        Display a notification message on screen

        Args:
            message: Text to display
            color: RGB color tuple
            duration: How long to show (seconds)
        """
        self.notification_message = message
        self.notification_color = color
        self.notification_timer = time.time() + duration
        print(f"📢 {message}")

    def check_game_state(self):
        """
        Check current game state and display appropriate notifications
        Returns: True if game is over, False otherwise
        """
        board = self.stockfish.board

        # Check for checkmate
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            self.show_notification(f"CHECKMATE! {winner} WINS!", (255, 215, 0), 999)
            self.game_over = True
            self.chess.winner = winner
            print(f"\n🏆 CHECKMATE! {winner} wins!")
            return True

        # Check for stalemate
        elif board.is_stalemate():
            self.show_notification("STALEMATE! Game is a DRAW!", (200, 200, 200), 999)
            self.game_over = True
            print(f"\n🤝 STALEMATE! Game is a draw!")
            return True

        # Check for insufficient material
        elif board.is_insufficient_material():
            self.show_notification("DRAW! Insufficient material", (200, 200, 200), 999)
            self.game_over = True
            print(f"\n🤝 DRAW! Insufficient material")
            return True

        # Check for fifty-move rule
        elif board.is_fifty_moves():
            self.show_notification("DRAW! Fifty-move rule", (200, 200, 200), 999)
            self.game_over = True
            print(f"\n🤝 DRAW! Fifty-move rule")
            return True

        # Check for threefold repetition
        elif board.is_repetition():
            self.show_notification("DRAW! Threefold repetition", (200, 200, 200), 999)
            self.game_over = True
            print(f"\n🤝 DRAW! Threefold repetition")
            return True

        # Check for check (but not checkmate)
        elif board.is_check():
            player = "Black" if board.turn == chess.BLACK else "White"
            self.show_notification(f"CHECK! {player} is in check!", (255, 100, 100), 3.0)
            print(f"\n⚠️  CHECK! {player} is in check!")
            return False

        return False

    def detect_human_move_from_occupancy(self, camera_occupancy):
        """
        HYBRID APPROACH: Detect move by comparing occupancy with Stockfish
        More robust - handles detection noise and looks for best matching legal move
        """
        print("\n🔍 Analyzing board occupancy vs CESA...")

        # Get Stockfish's current expected board state
        stockfish_board = self.stockfish.board

        # Find all occupancy differences
        differences = []

        for square_name, camera_state in camera_occupancy.items():
            try:
                square = chess.parse_square(square_name)
                stockfish_piece = stockfish_board.piece_at(square)

                # Convert Stockfish piece to occupancy
                if stockfish_piece is None:
                    expected_occupancy = 'empty'
                else:
                    expected_occupancy = 'white' if stockfish_piece.color == chess.WHITE else 'black'

                # Compare occupancy
                if camera_state != expected_occupancy:
                    differences.append({
                        'square': square_name,
                        'expected': expected_occupancy,
                        'detected': camera_state,
                        'stockfish_piece': stockfish_piece
                    })

            except Exception as e:
                print(f"   ⚠️  Error checking square {square_name}: {e}")
                continue

        print(f"   Found {len(differences)} occupancy differences")

        if len(differences) == 0:
            print("   ⚠️  No changes detected - board matches CESA")
            self.show_notification("No move detected - board unchanged", (255, 165, 0), 3.0)
            return None

        # Log differences
        for diff in differences:
            piece_name = str(diff['stockfish_piece']) if diff['stockfish_piece'] else 'none'
            print(f"   • {diff['square']}: expected {diff['expected']} ({piece_name}), detected {diff['detected']}")

        # Check if it's Black's turn
        if stockfish_board.turn != chess.BLACK:
            print(f"   ❌ Not Black's turn!")
            self.show_notification("ERROR: Not your turn!", (255, 50, 50), 3.0)
            return None

        # Get all legal Black moves
        legal_moves = list(stockfish_board.legal_moves)
        print(f"   Checking {len(legal_moves)} legal Black moves...")

        # Score each legal move based on how well it explains the differences
        move_scores = []

        for move in legal_moves:
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)

            # Calculate how many differences this move explains
            score = 0
            explained_squares = []

            # Check from_square: should go from black to empty
            from_diff = next((d for d in differences if d['square'] == from_square), None)
            if from_diff:
                if from_diff['expected'] == 'black' and from_diff['detected'] == 'empty':
                    score += 10  # Strong match
                    explained_squares.append(from_square)
                elif from_diff['detected'] == 'empty':
                    score += 5  # Weaker match
                    explained_squares.append(from_square)

            # Check to_square: should have black
            to_diff = next((d for d in differences if d['square'] == to_square), None)
            if to_diff:
                if to_diff['detected'] == 'black':
                    score += 10  # Strong match
                    explained_squares.append(to_square)

                    # Bonus if it was empty or had white (capture)
                    if to_diff['expected'] == 'empty':
                        score += 5  # Normal move
                    elif to_diff['expected'] == 'white':
                        score += 8  # Capture
                        print(f"   🎯 Potential capture at {to_square}")

            # Penalty for unexplained differences (but allow some noise)
            unexplained = len(differences) - len(explained_squares)
            penalty = max(0, unexplained - 4) * 2  # Allow up to 4 noise differences
            score -= penalty

            if score > 0:
                move_scores.append({
                    'move': move,
                    'score': score,
                    'from': from_square,
                    'to': to_square,
                    'explained': explained_squares
                })

        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x['score'], reverse=True)

        # Show top candidates
        if len(move_scores) > 0:
            print(f"\n   Top move candidates:")
            for i, candidate in enumerate(move_scores[:3]):
                print(f"   {i+1}. {candidate['from']}→{candidate['to']}: score={candidate['score']}, explains={candidate['explained']}")

        # Accept the best move if score is good enough
        if len(move_scores) > 0 and move_scores[0]['score'] >= 15:
            best = move_scores[0]
            print(f"\n   ✅ Best match: {best['from']} → {best['to']} (score: {best['score']})")

            piece = stockfish_board.piece_at(best['move'].from_square)
            piece_name = piece.symbol().lower() if piece else 'piece'
            piece_map = {
                'p': 'pawn', 'n': 'knight', 'b': 'bishop',
                'r': 'rook', 'q': 'queen', 'k': 'king'
            }

            return {
                'start': best['from'],
                'end': best['to'],
                'piece': piece_map.get(piece_name, 'piece')
            }

        print(f"\n   ❌ No confident move match (best score: {move_scores[0]['score'] if move_scores else 0})")
        self.show_notification("Could not identify legal move - try again", (255, 165, 0), 3.0)

        # Debug: show what legal moves were available
        if len(legal_moves) <= 20:
            legal_move_list = [chess.square_name(m.from_square) + chess.square_name(m.to_square) for m in legal_moves]
            print(f"   Legal Black moves: {legal_move_list}")

        return None

    def convert_tracker_grid_to_occupancy(self, tracker_grid):
        """
        Convert ChessTracker's 8x8 grid to occupancy dictionary

        Args:
            tracker_grid: 8x8 list of lists
                         Format: [['black', 'empty', ...], ...]
                         Row 0 = Rank 8, Row 7 = Rank 1

        Returns:
            Dictionary: {'a1': 'white', 'e2': 'white', 'e4': 'empty', ...}
        """
        occupancy = {}

        if not isinstance(tracker_grid, list) or len(tracker_grid) != 8:
            print(f"⚠️  Invalid tracker grid format")
            return {}

        files = 'abcdefgh'

        for row_idx, row in enumerate(tracker_grid):
            if not isinstance(row, list) or len(row) != 8:
                print(f"⚠️  Invalid row {row_idx}")
                continue

            rank = 8 - row_idx  # Row 0 = rank 8, Row 7 = rank 1

            for col_idx, piece_state in enumerate(row):
                # CRITICAL: Flip the file coordinates if board is rotated
                # Camera sees board from Black's perspective (rotated 180°)
                file = files[7 - col_idx]  # h,g,f,e,d,c,b,a instead of a,b,c,d,e,f,g,h
                square_name = f"{file}{rank}"

                # Normalize the state
                state = str(piece_state).lower().strip()

                if state in ['black', 'white', 'empty']:
                    occupancy[square_name] = state
                else:
                    print(f"⚠️  Unknown state '{piece_state}' at {square_name}")
                    occupancy[square_name] = 'empty'

        return occupancy

    def connect_camera(self):
        """Connect to Pi camera stream"""
        print(f"Connecting to camera at {self.stream_url}...")

        try:
            self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            if not self.cap.isOpened():
                print("❌ Could not connect to camera")
                return False

            print("✓ Connected to camera")
            self.camera_ready = True
            return True

        except Exception as e:
            print(f"❌ Camera connection error: {e}")
            return False

    def camera_frame_loop(self):
        """Background thread to continuously read camera frames"""
        print("Camera frame loop started")

        while self.running and self.camera_ready:
            if self.pause_camera_thread:
                time.sleep(0.1)
                continue

            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.annotated_frame = frame.copy()
                else:
                    time.sleep(0.1)

            except Exception as e:
                print(f"Frame read error: {e}")
                time.sleep(0.1)

    def capture_frame_with_retry(self):
        """Capture a single frame with retry logic and stream recovery"""
        max_retries = 3

        for retry in range(max_retries):
            self.pause_camera_thread = True
            time.sleep(0.2)

            # Flush buffer
            for _ in range(10):
                try:
                    self.cap.read()
                except:
                    pass

            time.sleep(0.3)

            # Try to capture frame
            try:
                ret, frame = self.cap.read()

                self.pause_camera_thread = False

                if ret and frame is not None and frame.size > 0:
                    return frame

            except Exception as e:
                # Stream error - try to recover
                if retry < max_retries - 1:
                    print(f"   Stream error, attempting recovery...")
                    self.pause_camera_thread = False

                    # Try to reconnect
                    try:
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(self.stream_url)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        time.sleep(0.5)
                    except:
                        pass
                else:
                    self.pause_camera_thread = False

        return None

    def show_retry_or_manual_prompt(self):
        """
        Show prompt asking user to retry camera or switch to manual mode.
        Returns: 'retry', 'manual', or None (quit)
        """
        print("\n" + "=" * 70)
        print("DETECTION FAILED")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Press 'R' - Try camera detection again")
        print("  2. Press 'M' - Use manual mode for THIS MOVE ONLY")
        print("  3. Press 'Q' - Quit")
        print("=" * 70)

        # Visual prompt in pygame window
        waiting = True

        while waiting and self.running:
            self.clock.tick(30)

            # Draw background
            self.screen.fill((40, 40, 40))

            # Draw camera feed on left
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = cv2.resize(self.current_frame, self.camera_display_size)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    self.screen.blit(frame_surface, (0, self.board_offset_y))

            # Draw board on right
            self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
            self.chess.draw_pieces()

            # Draw prompt
            font_title = pygame.font.SysFont("comicsansms", 36, bold=True)
            font_text = pygame.font.SysFont("comicsansms", 24)

            title = font_title.render("Detection Failed!", True, (255, 100, 100))
            title_rect = title.get_rect(center=(self.screen_width // 2, 100))
            self.screen.blit(title, title_rect)

            # Instructions
            y_pos = 200
            instructions = [
                "Camera detection failed",
                "",
                "Press 'R' - Retry camera",
                "Press 'M' - Use mouse for THIS MOVE",
                "Press 'Q' - Quit"
            ]

            for instruction in instructions:
                if instruction.startswith("Press"):
                    color = (255, 255, 100)
                elif instruction == "":
                    y_pos += 10
                    continue
                else:
                    color = (200, 200, 200)

                text = font_text.render(instruction, True, color)
                text_rect = text.get_rect(center=(self.screen_width // 2, y_pos))
                self.screen.blit(text, text_rect)
                y_pos += 40

            pygame.display.flip()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        print("\n✓ User chose: RETRY camera detection")
                        return 'retry'

                    elif event.key == pygame.K_m:
                        print("\n✓ User chose: MANUAL mode")
                        return 'manual'

                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        print("\n✓ User chose: QUIT")
                        self.running = False
                        return None

        return None

    def show_game_over_menu(self):
        """
        Show game over menu with options to play again or quit
        Returns: 'play_again' or 'quit'
        """
        print("\n" + "=" * 70)
        print("GAME OVER!")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Press 'P' - Play Again")
        print("  2. Press 'Q' - Quit")
        print("=" * 70)

        waiting = True

        while waiting and self.running:
            self.clock.tick(30)

            # Draw everything
            self.screen.fill((20, 20, 20))

            # Camera feed
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = cv2.resize(self.current_frame, self.camera_display_size)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    self.screen.blit(frame_surface, (0, self.board_offset_y))

            # Board
            self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
            self.chess.draw_pieces()

            # Game over overlay
            font_title = pygame.font.SysFont("comicsansms", 48, bold=True)
            font_text = pygame.font.SysFont("comicsansms", 28)
            font_small = pygame.font.SysFont("comicsansms", 22)

            # Winner announcement
            if self.chess.winner:
                winner_text = font_title.render(f"{self.chess.winner} Wins!", True, (255, 215, 0))
            else:
                winner_text = font_title.render("Game Over!", True, (200, 200, 200))

            winner_rect = winner_text.get_rect(center=(self.screen_width // 2, 150))

            # Background box
            bg_rect = pygame.Rect(0, 0, 600, 400)
            bg_rect.center = (self.screen_width // 2, self.screen_height // 2)
            pygame.draw.rect(self.screen, (30, 30, 30), bg_rect)
            pygame.draw.rect(self.screen, (255, 215, 0), bg_rect, 4)

            self.screen.blit(winner_text, winner_rect)

            # Move count
            moves_text = font_text.render(f"Total Moves: {self.move_count // 2}", True, (200, 200, 200))
            moves_rect = moves_text.get_rect(center=(self.screen_width // 2, 250))
            self.screen.blit(moves_text, moves_rect)

            # Options
            y_pos = 320
            option1 = font_small.render("Press 'P' - Play Again", True, (100, 255, 100))
            option1_rect = option1.get_rect(center=(self.screen_width // 2, y_pos))
            self.screen.blit(option1, option1_rect)

            option2 = font_small.render("Press 'Q' - Quit", True, (255, 100, 100))
            option2_rect = option2.get_rect(center=(self.screen_width // 2, y_pos + 40))
            self.screen.blit(option2, option2_rect)

            pygame.display.flip()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return 'quit'

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        print("\n✓ User chose: PLAY AGAIN")
                        return 'play_again'

                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        print("\n✓ User chose: QUIT")
                        self.running = False
                        return 'quit'

        return 'quit'

    def reset_game(self):
        """Reset the game state for a new game"""
        print("\n" + "=" * 70)
        print("RESETTING GAME...")
        print("=" * 70)

        # Reset Stockfish engine
        self.stockfish.board.reset()

        # Reset game state
        self.move_count = 0
        self.move_history = []
        self.waiting_for_human = False
        self.game_over = False
        self.in_manual_mode = False
        self.manual_mode_temporary = False
        self.game_started = False

        # Reset chess GUI
        self.chess.reset_game()

        # Clear notifications
        self.notification_message = None
        self.notification_timer = 0

        print("✓ Game reset complete!")
        print("=" * 70)

    def wait_for_move_capture(self):
        """
        VIDEO STREAM MODE: Wait for SPACE, then analyze live video stream
        until perfect 64/64 detection is found
        """
        print("\n" + "=" * 70)
        print(f"YOUR TURN - Move #{(self.move_count // 2) + 1}")
        print("=" * 70)

        if self.in_manual_mode:
            print("MANUAL MODE: Use mouse to move pieces")
            return 'manual'

        # Get Stockfish's last move for display
        stockfish_last_move = None
        if len(self.stockfish.board.move_stack) > 0:
            last_move = self.stockfish.board.peek()
            stockfish_last_move = {
                'start': chess.square_name(last_move.from_square),
                'end': chess.square_name(last_move.to_square)
            }

        print("📋 Instructions:")
        if stockfish_last_move:
            print(f"   1. Move WHITE piece: {stockfish_last_move['start']} → {stockfish_last_move['end']}")
        print(f"   2. Make YOUR BLACK move")
        print(f"   3. Press SPACE when both moves are complete")

        # Wait for SPACE
        waiting = True
        while waiting and self.running:
            self.clock.tick(30)

            self.screen.fill((0, 0, 0))

            # Draw camera feed
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = cv2.resize(self.current_frame, self.camera_display_size)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    self.screen.blit(frame_surface, (0, self.board_offset_y))

            # Draw board
            self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
            self.chess.draw_pieces()

            # Instructions
            font = pygame.font.SysFont("comicsansms", 20)
            y = 10

            if stockfish_last_move:
                text1 = font.render(f"1. Move White: {stockfish_last_move['start']}→{stockfish_last_move['end']}",
                                  True, (100, 200, 255))
                self.screen.blit(text1, (10, y))
                y += 25

            text2 = font.render("2. Make your Black move", True, (100, 255, 100))
            self.screen.blit(text2, (10, y))
            y += 25

            text3 = font.render("3. Press SPACE to start detection", True, (255, 255, 100))
            self.screen.blit(text3, (10, y))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        self.running = False
                        return None

        if not self.running:
            return None

        # Countdown
        for i in range(3, 0, -1):
            self.screen.fill((0, 0, 0))

            with self.frame_lock:
                if self.current_frame is not None:
                    frame = cv2.resize(self.current_frame, self.camera_display_size)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    self.screen.blit(frame_surface, (0, self.board_offset_y))

            self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
            self.chess.draw_pieces()

            font_big = pygame.font.SysFont("comicsansms", 72)
            countdown_text = font_big.render(f"{i}", True, (0, 255, 0))
            text_rect = countdown_text.get_rect(center=(320, 400))
            self.screen.blit(countdown_text, text_rect)

            pygame.display.flip()
            time.sleep(1)

        # VIDEO STREAM ANALYSIS MODE
        print(f"\n📹 Analyzing video stream for perfect detection...")

        max_duration = 15  # Maximum 15 seconds of analysis
        start_time = time.time()
        frames_analyzed = 0
        best_count = 0

        while time.time() - start_time < max_duration:
            # Get current frame from video stream
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.033)
                    continue
                frame = self.current_frame.copy()

            frames_analyzed += 1

            # Save frame temporarily
            temp_path = f"temp_video_frame.jpg"
            cv2.imwrite(temp_path, frame)

            # Analyze frame
            status_new, annotated, count = self.tracker.detect_board_from_image(temp_path)

            # Track best
            if count > best_count:
                best_count = count

            # Progress update every 30 frames (~1 second)
            if frames_analyzed % 30 == 0:
                elapsed = time.time() - start_time
                print(f"   📊 {frames_analyzed} frames analyzed ({elapsed:.1f}s), best: {best_count}/64")

            # Perfect detection found!
            if count == 64 and status_new is not None:
                elapsed = time.time() - start_time
                print(f"✅ Perfect detection! ({frames_analyzed} frames, {elapsed:.1f}s)")

                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass

                return {
                    'status': status_new,
                    'annotated': annotated,
                    'count': count,
                    'frame': frame
                }

            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass

            # Small delay to achieve ~30 FPS analysis
            time.sleep(0.033)

        # Failed to find perfect detection
        elapsed = time.time() - start_time
        print(f"\n❌ No perfect detection in {frames_analyzed} frames ({elapsed:.1f}s)")
        print(f"   Best detection: {best_count}/64 squares")

        # Prompt for retry or manual
        choice = self.show_retry_or_manual_prompt()

        if choice == 'retry':
            return self.wait_for_move_capture()
        elif choice == 'manual':
            self.in_manual_mode = True
            self.manual_mode_temporary = True
            print("\n🖱️  MANUAL MODE ACTIVATED (for this move only)")
            return 'manual'
        else:
            return None

    def handle_manual_move(self):
        """Handle mouse-based piece movement"""
        if not self.utils.left_click_event():
            return False

        mouse_pos = self.utils.get_mouse_event()

        # Check if click is on board area
        if mouse_pos[0] < self.board_offset_x:
            return False

        # Convert to board coordinates
        board_x = (mouse_pos[0] - self.board_offset_x) // self.square_length
        board_y = (mouse_pos[1] - self.board_offset_y) // self.square_length

        if board_x < 0 or board_x > 7 or board_y < 0 or board_y > 7:
            return False

        # Track if piece was selected before this click
        was_piece_selected = self.chess.selected_piece is not None
        selected_from = None

        if was_piece_selected:
            selected_from = (self.chess.selected_pos[0], self.chess.selected_pos[1])

        # Let chess handle the move
        self.chess.move_piece_manual(board_x, board_y)

        # Check if a move was just completed
        move_was_completed = was_piece_selected and self.chess.selected_piece is None

        if move_was_completed:
            end_col = chr(ord('a') + board_x)
            end_row = 8 - board_y

            if selected_from:
                start_col, start_row = selected_from

                move_notation = f"{start_col}{start_row}{end_col}{end_row}"

                print(f"\n✅ Manual move: {start_col}{start_row} → {end_col}{end_row}")

                # Apply to Stockfish
                if self.stockfish.apply_move(move_notation):
                    print(f"   ✓ Synced with CESA")
                    self.move_count += 1
                    self.move_history.append(f"{start_col}{start_row}-{end_col}{end_row}")

                    # Check game state
                    self.check_game_state()

                    if self.manual_mode_temporary:
                        self.in_manual_mode = False
                        self.manual_mode_temporary = False
                        self.waiting_for_human = False

                        print("📷 Returning to CAMERA MODE")
                else:
                    print(f"   ❌ Illegal move!")
                    self.show_notification("ILLEGAL MOVE! Try again.", (255, 50, 50), 3.0)

        return True

    def make_stockfish_move(self):
        """Get and execute Stockfish's move"""
        print("\n" + "=" * 70)
        print(f"CESA (WHITE) THINKING - Move #{(self.move_count // 2) + 1}")
        print("=" * 70)

        # Show "thinking" message on GUI
        self.draw_thinking_message()

        # Get Stockfish's move
        move_tuple = self.stockfish.get_best_move()

        if move_tuple is None:
            print("⚠️  CESA could not generate a move (game over?)")
            return False

        start_square, end_square = move_tuple

        print(f"🤖 CESA plays: {start_square} → {end_square}")

        # Create move dict for chess bridge
        move_dict = {
            'start': start_square,
            'end': end_square,
            'piece': 'unknown'
        }

        # Apply to GUI
        success = self.chess.apply_camera_move(move_dict)

        if success:
            self.move_count += 1
            self.move_history.append(f"{start_square}-{end_square}")
            self.waiting_for_human = True

            # Show Stockfish's move
            self.show_notification(f"CESA: {start_square}→{end_square}", (100, 200, 255), 5.0)

            # Check game state (for checkmate/check by Stockfish)
            self.check_game_state()

            print(f"✅ CESA's move applied to GUI")
            print("\n♟️  YOUR TURN (BLACK)")
            print(f"   1. Move WHITE piece on board: {start_square} → {end_square}")
            print(f"   2. Make your BLACK move")
            print(f"   3. Press SPACE when both done")

            time.sleep(2)

            return True
        else:
            print("❌ Failed to apply CESA's move to GUI")
            return False

    def draw_thinking_message(self):
        """Draw 'CESA is thinking...' message"""
        self.screen.fill((20, 20, 20))

        # Camera feed
        with self.frame_lock:
            if self.current_frame is not None:
                frame = cv2.resize(self.current_frame, self.camera_display_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                self.screen.blit(frame_surface, (0, self.board_offset_y))

        # Board
        self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
        self.chess.draw_pieces()

        # Thinking message
        font = pygame.font.SysFont("comicsansms", 36, bold=True)
        text = font.render("🤖 CESA is thinking...", True, (100, 200, 255))
        text_rect = text.get_rect(center=(self.screen_width // 2, 20))
        self.screen.blit(text, text_rect)

        pygame.display.flip()

    def draw_notification(self):
        """Draw notification message if active"""
        if self.notification_message and time.time() < self.notification_timer:
            font = pygame.font.SysFont("comicsansms", 32, bold=True)
            text = font.render(self.notification_message, True, self.notification_color)

            # Draw with background
            text_rect = text.get_rect(center=(self.screen_width // 2, 720))
            bg_rect = text_rect.inflate(40, 20)

            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            pygame.draw.rect(self.screen, self.notification_color, bg_rect, 3)
            self.screen.blit(text, text_rect)

    def draw_move_history(self):
        """Draw recent move history on screen"""
        if len(self.move_history) == 0:
            return

        font = pygame.font.SysFont("monospace", 16)
        y_offset = self.screen_height - 100

        # Show last 3 moves
        recent_moves = self.move_history[-6:]
        for i, move in enumerate(recent_moves):
            move_num = len(self.move_history) - len(recent_moves) + i + 1
            color_label = "C" if i % 2 == 0 else "H"  # C=CESA, H=Human
            text = font.render(f"{move_num}. {color_label}: {move}", True, (180, 180, 180))
            self.screen.blit(text, (self.screen_width - 200, y_offset + i * 20))

    def run(self):
        """Main game loop with replay functionality"""

        # Connect camera
        if not self.connect_camera():
            print("❌ Failed to connect camera")
            return

        # Start camera thread
        camera_thread = threading.Thread(target=self.camera_frame_loop, daemon=True)
        camera_thread.start()
        time.sleep(1)

        # Main game loop - can replay multiple games
        while self.running:
            print("\n" + "=" * 70)
            print("GAME STARTED!")
            print("=" * 70)
            print("♟️  You are BLACK (Human), CESA is WHITE")
            print("=" * 70)

            # CESA makes first move (White)
            print("\n🤖 CESA (White) will make the opening move...")
            time.sleep(1)

            if not self.make_stockfish_move():
                print("❌ Failed to make opening move")
                break

            self.game_started = True

            # Game loop
            while self.running and not self.game_over:
                self.clock.tick(30)

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            self.running = False

                # Manual mode - handle mouse
                if self.in_manual_mode:
                    self.handle_manual_move()

                # Draw everything
                self.screen.fill((20, 20, 20))

                # Camera feed
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = cv2.resize(self.current_frame, self.camera_display_size)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                        self.screen.blit(frame_surface, (0, self.board_offset_y))

                # Board
                self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
                self.chess.draw_pieces()

                # Turn indicator
                font = pygame.font.SysFont("comicsansms", 24, bold=True)
                if self.game_over:
                    # Game over - show in notification instead
                    pass
                elif self.waiting_for_human:
                    turn_text = "YOUR TURN (BLACK)"
                    color = (100, 255, 100)
                    text = font.render(turn_text, True, color)
                    self.screen.blit(text, (10, 10))
                else:
                    turn_text = "CESA's TURN (WHITE)"
                    color = (100, 200, 255)
                    text = font.render(turn_text, True, color)
                    self.screen.blit(text, (10, 10))

                # Mode indicator
                mode_font = pygame.font.SysFont("comicsansms", 18)
                if self.in_manual_mode:
                    mode_text = "MODE: MANUAL"
                    mode_color = (255, 165, 0)
                else:
                    mode_text = "MODE: VIDEO STREAM"
                    mode_color = (0, 200, 0)

                mode_surface = mode_font.render(mode_text, True, mode_color)
                self.screen.blit(mode_surface, (10, 40))

                # Move counter
                move_text = font.render(f"Moves: {self.move_count // 2}", True, (255, 255, 255))
                self.screen.blit(move_text, (self.screen_width - 150, 10))

                # Move history
                self.draw_move_history()

                # Notifications (check, checkmate, etc.)
                self.draw_notification()

                # Winner overlay (legacy - now using notifications)
                if self.chess.winner:
                    font_big = pygame.font.SysFont("comicsansms", 48, bold=True)
                    winner_text = font_big.render(f"{self.chess.winner} Wins!", True, (255, 215, 0))
                    winner_rect = winner_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
                    bg_rect = winner_rect.inflate(40, 20)
                    pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
                    pygame.draw.rect(self.screen, (255, 215, 0), bg_rect, 3)
                    self.screen.blit(winner_text, winner_rect)

                pygame.display.flip()

                # Game logic: alternate between human and CESA
                if not self.game_over and self.game_started:
                    if self.waiting_for_human and not self.in_manual_mode:
                        # Wait for human to make both moves
                        result = self.wait_for_move_capture()

                        if result is None:
                            break
                        elif result == 'manual':
                            continue
                        elif isinstance(result, dict):
                            # Got board state from camera
                            tracker_grid = result['status']

                            if tracker_grid is None:
                                print("⚠️  Invalid board detection")
                                continue

                            # Convert grid to occupancy
                            camera_occupancy = self.convert_tracker_grid_to_occupancy(tracker_grid)

                            if len(camera_occupancy) != 64:
                                print(f"⚠️  Incomplete occupancy data: {len(camera_occupancy)}/64 squares")
                                continue

                            print(f"✓ Converted to occupancy: {len(camera_occupancy)} squares")

                            # HYBRID: Detect move from occupancy differences
                            move = self.detect_human_move_from_occupancy(camera_occupancy)

                            if move and 'start' in move and 'end' in move:
                                # Apply to GUI
                                success = self.chess.apply_camera_move(move)

                                if success:
                                    self.move_count += 1
                                    self.move_history.append(f"{move['start']}-{move['end']}")
                                    print(f"✅ BLACK Move {self.move_count // 2}: {move['start']} → {move['end']}")

                                    # Show move notification
                                    self.show_notification(f"You: {move['start']}→{move['end']}", (100, 255, 100), 3.0)

                                    # Apply to Stockfish's internal board
                                    move_uci = move['start'] + move['end']
                                    if self.stockfish.apply_move(move_uci):
                                        print(f"   ✓ Synced with CESA")
                                    else:
                                        print(f"   ❌ Failed to sync with CESA")
                                        self.show_notification("ERROR: Failed to sync move", (255, 50, 50), 3.0)
                                        continue

                                    # Check game state after human move
                                    if self.check_game_state():
                                        # Game over
                                        continue

                                    # CESA's turn
                                    self.waiting_for_human = False
                                else:
                                    print("⚠️  Failed to apply move to GUI")
                                    self.show_notification("Failed to apply move", (255, 165, 0), 3.0)
                            else:
                                print("⚠️  No valid move detected - try again")

                    elif not self.waiting_for_human:
                        # CESA's turn
                        if self.make_stockfish_move():
                            self.waiting_for_human = True
                        else:
                            print("❌ CESA move failed")
                            break

            # Game ended - show game over menu
            if self.game_over:
                print("\n" + "=" * 70)
                print("GAME ENDED")
                print(f"Total moves: {self.move_count // 2}")
                if self.chess.winner:
                    print(f"Winner: {self.chess.winner}")
                print("\nMove History:")
                for i, move in enumerate(self.move_history):
                    print(f"  {i+1}. {move}")
                print("=" * 70)

                # Show game over menu
                choice = self.show_game_over_menu()

                if choice == 'play_again':
                    self.reset_game()
                    continue  # Start new game
                else:
                    break  # Exit game loop

            # If we broke out without game over (user quit), exit
            if not self.game_over:
                break

        # Final cleanup
        if self.cap:
            self.cap.release()
        self.stockfish.close()
        pygame.quit()

        print("\n✓ Thanks for playing CESA!")


def main():
    """Main entry point"""
    print("=" * 70)
    print("CESA: Cognitive Embodied Strategic Agent")
    print("=" * 70)
    print("\n📋 Features:")
    print("   • CESA AI with adjustable difficulty (0-20)")
    print("   • VIDEO STREAM detection - fast & efficient")
    print("   • Game state notifications (Check, Checkmate, Stalemate)")
    print("   • Illegal move prevention")
    print("   • Move history tracking")
    print("   • Manual mouse fallback")
    print("   • Play Again feature")
    print("\n🎮 Controls:")
    print("   • SPACE - Start video stream detection")
    print("   • R - Retry detection")
    print("   • M - Switch to manual mode")
    print("   • P - Play Again (after game ends)")
    print("   • Q/ESC - Quit")
    print("=" * 70)

    # Set Stockfish skill level
    skill = input("\nEnter CESA skill level (0-20, default 5): ").strip()
    if skill.isdigit():
        skill_level = int(skill)
        skill_level = max(0, min(20, skill_level))
    else:
        skill_level = 5

    print(f"\n🤖 Starting game with CESA skill level {skill_level}...")

    system = HybridChessSystem(
        STREAM_URL,
        stockfish_path="C:\\Program Files\\stockfish\\stockfish\\stockfish-windows-x86-64-avx2.exe",
        skill_level=skill_level
    )

    system.run()


if __name__ == "__main__":
    main()