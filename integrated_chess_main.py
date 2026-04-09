"""
Integrated Chess System
Combines YOLOv11 camera tracking with Pygame GUI
Displays both camera feed and virtual board side-by-side
"""

import cv2
import pygame
import threading
import queue
import numpy as np
from chess_tracker import ChessTracker
from chess_camera_bridge import ChessCameraBridge
from piece import Piece
import time
import os

# Replace with your Pi's IP
PI_IP = "10.223.89.228"
STREAM_URL = f"http://{PI_IP}:5000/video_feed"

class IntegratedChessSystem:
    def __init__(self, stream_url=STREAM_URL):
        """Initialize integrated chess system"""
        self.stream_url = stream_url
        
        # Initialize camera tracker
        self.tracker = ChessTracker()
        
        # Queue for passing moves from camera thread to GUI thread
        self.move_queue = queue.Queue()
        
        # Threading controls
        self.running = True
        self.camera_ready = False
        self.game_started = False
        self.waiting_for_move = False
        self.pause_camera_thread = False  # NEW: Flag to pause background reading

        # Camera capture
        self.cap = None
        self.current_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()

        # Board state tracking
        self.current_board_status = None
        self.move_count = 0

        # Pygame/GUI setup
        self.screen_width = 1280  # 640 for camera + 640 for chess board
        self.screen_height = 750
        self.camera_display_size = (640, 640)
        self.board_offset_x = 640  # Board starts after camera feed
        self.board_offset_y = 50

        # Initialize Pygame
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        pygame.display.set_caption("Chess - Camera + Board")

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

        print("✓ Pygame initialized")

    def connect_camera(self):
        """Connect to Pi camera stream"""
        print(f"Connecting to camera at {self.stream_url}...")
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("❌ Failed to connect to camera")
            return False

        print("✓ Camera connected")
        self.camera_ready = True
        return True

    def capture_frame_with_retry(self, num_retries=5):
        """Capture a frame with retry logic and error handling"""
        # PAUSE background thread to prevent read conflicts
        self.pause_camera_thread = True
        time.sleep(0.2)  # Give thread time to pause

        # Flush buffer thoroughly
        for _ in range(10):
            try:
                ret, _ = self.cap.read()
            except:
                pass

        time.sleep(0.3)

        # Try to get a valid frame
        frame = None
        for attempt in range(num_retries):
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    break
            except cv2.error as e:
                # OpenCV error - try to recover
                if attempt == num_retries - 1:
                    # Last attempt - try reconnecting
                    print(f"   ⚠️  Reconnecting camera...")
                    try:
                        self.cap.release()
                    except:
                        pass
                    time.sleep(1)
                    if self.connect_camera():
                        # One more try after reconnect
                        try:
                            ret, frame = self.cap.read()
                            if ret and frame is not None and frame.size > 0:
                                break
                        except:
                            pass
                time.sleep(0.2)
                continue
            except Exception as e:
                time.sleep(0.2)
                continue

        # RESUME background thread
        self.pause_camera_thread = False

        return frame

    def camera_thread_func(self):
        """Thread that handles camera capture and continuous feed"""
        print("📷 Camera thread started")

        if not self.connect_camera():
            return

        consecutive_errors = 0
        max_consecutive_errors = 5

        # Main camera loop - continuously update frame
        while self.running:
            # PAUSE if main thread is capturing
            if self.pause_camera_thread:
                time.sleep(0.1)
                continue

            try:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"⚠️  Camera stream unstable, reconnecting...")
                        self.cap.release()
                        time.sleep(1)
                        if not self.connect_camera():
                            print("❌ Failed to reconnect to camera")
                            break
                        consecutive_errors = 0
                    continue

                # Reset error counter on successful read
                consecutive_errors = 0

                # Resize to fit display
                frame_resized = cv2.resize(frame, self.camera_display_size)

                with self.frame_lock:
                    self.current_frame = frame
                    self.annotated_frame = frame_resized

            except cv2.error as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"⚠️  OpenCV error, reconnecting camera...")
                    try:
                        self.cap.release()
                    except:
                        pass
                    time.sleep(1)
                    if not self.connect_camera():
                        print("❌ Failed to reconnect to camera")
                        break
                    consecutive_errors = 0
                else:
                    time.sleep(0.1)
                continue
            except Exception as e:
                print(f"⚠️  Camera thread error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(0.1)
                continue

            time.sleep(0.033)  # ~30 FPS

        try:
            self.cap.release()
        except:
            pass
        print("📷 Camera thread stopped")

    def capture_initial_position(self):
        """Capture initial board position with retry logic"""
        print("\n" + "=" * 70)
        print("CAPTURING INITIAL POSITION")
        print("=" * 70)
        print("Position your chess pieces on the board")
        print("Press SPACE when ready to capture")
        print("Press Q to quit")

        waiting = True
        message = "Press SPACE to capture initial position | Q to quit"

        while waiting and self.running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_q:
                        self.running = False
                        return False

            # Draw interface
            self.screen.fill((0, 0, 0))

            # Draw camera feed
            with self.frame_lock:
                if self.annotated_frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(self.annotated_frame, cv2.COLOR_BGR2RGB)
                    # Convert to pygame surface
                    frame_surface = pygame.surfarray.make_surface(
                        np.transpose(frame_rgb, (1, 0, 2))
                    )
                    self.screen.blit(frame_surface, (0, 50))

            # Draw instructions
            font = pygame.font.SysFont("comicsansms", 24)
            text = font.render(message, True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

            pygame.display.flip()
            self.clock.tick(30)

        if not self.running:
            return False

        # Countdown
        for i in range(3, 0, -1):
            self.screen.fill((0, 0, 0))

            with self.frame_lock:
                if self.annotated_frame is not None:
                    frame_rgb = cv2.cvtColor(self.annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(
                        np.transpose(frame_rgb, (1, 0, 2))
                    )
                    self.screen.blit(frame_surface, (0, 50))

            font_big = pygame.font.SysFont("comicsansms", 72)
            countdown_text = font_big.render(f"Capturing in {i}...", True, (0, 255, 0))
            text_rect = countdown_text.get_rect(center=(320, 400))
            self.screen.blit(countdown_text, text_rect)

            pygame.display.flip()
            time.sleep(1)

        # Try capturing with retries
        max_attempts = 10
        best_count = 0
        best_result = None

        print(f"\n📸 Capturing initial position (up to {max_attempts} attempts)...")

        for attempt in range(1, max_attempts + 1):
            print(f"   Attempt {attempt}/{max_attempts}...", end=" ")

            frame = self.capture_frame_with_retry()
            if frame is None:
                print("❌ Failed to capture frame")
                continue

            # Save and detect
            temp_path = f"temp_initial_attempt_{attempt}.jpg"
            cv2.imwrite(temp_path, frame)

            status, annotated, count = self.tracker.detect_board_from_image(temp_path)

            print(f"Detected {count}/64 squares")

            # If we got a perfect detection, use it immediately
            if count == 64 and status is not None:
                print(f"✅ Perfect detection on attempt {attempt}!")
                self.current_board_status = status
                self.tracker.previous_board_status = status

                if annotated is not None:
                    cv2.imwrite("initial_board_detected.jpg", annotated)
                    print("💾 Saved: initial_board_detected.jpg")

                self.game_started = True

                # Clean up temp files
                for i in range(1, attempt + 1):
                    try:
                        os.remove(f"temp_initial_attempt_{i}.jpg")
                    except:
                        pass

                return True

            # Track best attempt
            if count > best_count:
                best_count = count
                best_result = {
                    'status': status,
                    'annotated': annotated,
                    'count': count,
                    'attempt': attempt
                }

            # Longer delay between attempts to give camera time to stabilize
            time.sleep(1.0)

        # Clean up temp files
        for i in range(1, max_attempts + 1):
            try:
                os.remove(f"temp_initial_attempt_{i}.jpg")
            except:
                pass

        # If we got at least 60 squares on best attempt, use it
        if best_result and best_count >= 60:
            print(f"\n⚠️  Using best result: {best_count}/64 squares (attempt {best_result['attempt']})")
            self.current_board_status = best_result['status']
            self.tracker.previous_board_status = best_result['status']

            if best_result['annotated'] is not None:
                cv2.imwrite("initial_board_detected.jpg", best_result['annotated'])
                print("💾 Saved: initial_board_detected.jpg")

            self.game_started = True
            return True

        # Failed all attempts
        print(f"\n❌ Failed to capture board after {max_attempts} attempts")
        print(f"   Best result: {best_count}/64 squares")
        print("\n💡 Suggestions:")
        print("   • Improve lighting (especially on black pieces)")
        print("   • Ensure entire board is visible")
        print("   • Position camera directly above board")
        print("   • Reduce shadows")
        return False

    def wait_for_move_capture(self):
        """Wait for player to make a move, then capture it with retry logic"""
        print("\n" + "=" * 70)
        print(f"WAITING FOR MOVE #{self.move_count + 1}")
        print("=" * 70)
        print("Make your move on the physical board")
        print("Press SPACE when move is complete")
        print("Press R to retry | Q to quit")

        waiting = True
        message = f"Move #{self.move_count + 1} | SPACE: Capture move | R: Retry | Q: Quit"

        while waiting and self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_r:
                        return 'retry'
                    elif event.key == pygame.K_q:
                        self.running = False
                        return None

            # Draw interface
            self.screen.fill((0, 0, 0))

            # Draw camera feed
            with self.frame_lock:
                if self.annotated_frame is not None:
                    frame_rgb = cv2.cvtColor(self.annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(
                        np.transpose(frame_rgb, (1, 0, 2))
                    )
                    self.screen.blit(frame_surface, (0, 50))

            # Draw chess board
            self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
            self.chess.draw_pieces()

            # Draw turn indicator
            turn_font = pygame.font.SysFont("comicsansms", 20)
            if self.chess.turn["black"] == 0 and self.chess.turn["white"] == 0:
                turn_text = turn_font.render("Waiting for first move...", True, (255, 255, 0))
            elif self.chess.turn["black"]:
                turn_text = turn_font.render("Turn: Black", True, (255, 255, 255))
            else:
                turn_text = turn_font.render("Turn: White", True, (255, 255, 255))
            self.screen.blit(turn_text, (self.board_offset_x + 200, 10))

            # Draw instructions
            font = pygame.font.SysFont("comicsansms", 18)
            text = font.render(message, True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

            pygame.display.flip()
            self.clock.tick(30)

        if not self.running:
            return None

        # Countdown
        for i in range(3, 0, -1):
            self.screen.fill((0, 0, 0))

            with self.frame_lock:
                if self.annotated_frame is not None:
                    frame_rgb = cv2.cvtColor(self.annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(
                        np.transpose(frame_rgb, (1, 0, 2))
                    )
                    self.screen.blit(frame_surface, (0, 50))

            self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
            self.chess.draw_pieces()

            font_big = pygame.font.SysFont("comicsansms", 72)
            countdown_text = font_big.render(f"{i}", True, (0, 255, 0))
            text_rect = countdown_text.get_rect(center=(320, 400))
            self.screen.blit(countdown_text, text_rect)

            pygame.display.flip()
            time.sleep(1)

        # Try capturing with retries
        max_attempts = 10
        best_count = 0
        best_result = None

        print(f"\n📸 Capturing move (up to {max_attempts} attempts)...")

        for attempt in range(1, max_attempts + 1):
            print(f"   Attempt {attempt}/{max_attempts}...", end=" ")

            frame = self.capture_frame_with_retry()
            if frame is None:
                print("❌ Failed to capture frame")
                continue

            # Save and detect
            temp_path = f"temp_move_{self.move_count + 1}_attempt_{attempt}.jpg"
            cv2.imwrite(temp_path, frame)

            status_new, annotated, count = self.tracker.detect_board_from_image(temp_path)

            print(f"Detected {count}/64 squares")

            # If we got a perfect detection, use it immediately
            if count == 64 and status_new is not None:
                print(f"✅ Perfect detection on attempt {attempt}!")

                # Clean up temp files
                for i in range(1, attempt + 1):
                    try:
                        os.remove(f"temp_move_{self.move_count + 1}_attempt_{i}.jpg")
                    except:
                        pass

                return {
                    'status': status_new,
                    'annotated': annotated,
                    'count': count,
                    'frame': frame
                }

            # Track best attempt
            if count > best_count:
                best_count = count
                best_result = {
                    'status': status_new,
                    'annotated': annotated,
                    'count': count,
                    'frame': frame,
                    'attempt': attempt
                }

            # Longer delay between attempts to give camera time to stabilize
            time.sleep(1.0)

        # Clean up temp files
        for i in range(1, max_attempts + 1):
            try:
                os.remove(f"temp_move_{self.move_count + 1}_attempt_{i}.jpg")
            except:
                pass

        # If we got at least 60 squares on best attempt, use it
        if best_result and best_count >= 60:
            print(f"\n⚠️  Using best result: {best_count}/64 squares (attempt {best_result['attempt']})")
            return best_result

        # Failed all attempts
        print(f"\n❌ Failed to capture move after {max_attempts} attempts")
        print(f"   Best result: {best_count}/64 squares")

        return {
            'status': None,
            'annotated': None,
            'count': 0,
            'frame': None
        }

    def run(self):
        """Main game loop"""
        # Start camera thread
        camera_thread = threading.Thread(target=self.camera_thread_func, daemon=True)
        camera_thread.start()

        # Wait for camera to be ready
        while not self.camera_ready and self.running:
            self.screen.fill((0, 0, 0))
            font = pygame.font.SysFont("comicsansms", 24)
            text = font.render("Connecting to camera...", True, (255, 255, 255))
            self.screen.blit(text, (400, 350))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            time.sleep(0.1)

        if not self.running:
            return

        # Capture initial position
        if not self.capture_initial_position():
            print("\n❌ Failed to capture initial position")
            self.running = False
            return

        print("\n" + "=" * 70)
        print("GAME STARTED!")
        print("=" * 70)

        # Main game loop
        try:
            while self.running and not self.chess.winner:
                # Wait for move
                result = self.wait_for_move_capture()

                if result is None:
                    break

                if result == 'retry':
                    print("🔄 Retrying initial position...")
                    if self.capture_initial_position():
                        continue
                    else:
                        break

                status_new = result.get('status')
                count = result.get('count', 0)
                annotated = result.get('annotated')

                if status_new is None or count < 60:
                    print(f"❌ Failed to detect board (detected {count}/64 squares)")
                    print("Press SPACE to retry")

                    # Show error and wait
                    waiting = True
                    while waiting and self.running:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.running = False
                                waiting = False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    waiting = False

                        self.screen.fill((0, 0, 0))
                        font = pygame.font.SysFont("comicsansms", 24)
                        text = font.render(f"Detection failed ({count}/64) - Press SPACE to retry",
                                         True, (255, 0, 0))
                        self.screen.blit(text, (200, 350))
                        pygame.display.flip()
                        self.clock.tick(30)

                    continue

                print(f"✅ Detected {count}/64 squares")

                # Detect move
                try:
                    move = self.tracker.detect_move(self.current_board_status, status_new)
                except Exception as e:
                    print(f"❌ Error detecting move: {e}")
                    continue

                if 'start' in move and 'end' in move:
                    self.move_count += 1
                    print(f"\n📍 MOVE #{self.move_count} DETECTED")
                    print(f"   {move.get('piece', 'Piece').capitalize()}: {move['start']} → {move['end']}")

                    if 'eliminated' in move:
                        print(f"   🎯 Captured: {move['eliminated']}")

                    # Apply move to GUI
                    success = self.chess.apply_camera_move(move)

                    if success:
                        self.current_board_status = status_new

                        # Save annotated image
                        if annotated is not None:
                            cv2.imwrite(f"move_{self.move_count}.jpg", annotated)
                            print(f"💾 Saved: move_{self.move_count}.jpg")
                    else:
                        print("⚠️ Move validation failed, but continuing...")
                        self.current_board_status = status_new

                else:
                    print("\n⚠️ NO MOVE DETECTED - Press SPACE to continue")

                    waiting = True
                    while waiting and self.running:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.running = False
                                waiting = False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    waiting = False

                        self.screen.fill((0, 0, 0))
                        with self.frame_lock:
                            if self.annotated_frame is not None:
                                frame_rgb = cv2.cvtColor(self.annotated_frame, cv2.COLOR_BGR2RGB)
                                frame_surface = pygame.surfarray.make_surface(
                                    np.transpose(frame_rgb, (1, 0, 2))
                                )
                                self.screen.blit(frame_surface, (0, 50))

                        self.screen.blit(self.board_img, (self.board_offset_x, self.board_offset_y))
                        self.chess.draw_pieces()

                        font = pygame.font.SysFont("comicsansms", 24)
                        text = font.render("No move detected - Press SPACE to continue",
                                         True, (255, 255, 0))
                        self.screen.blit(text, (200, 10))
                        pygame.display.flip()
                        self.clock.tick(30)

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            print("\n" + "=" * 70)
            print("GAME SUMMARY")
            print("=" * 70)
            print(f"Total moves tracked: {self.move_count}")
            if self.chess.winner:
                print(f"🏆 Winner: {self.chess.winner}")
            print("=" * 70)

            self.running = False
            time.sleep(0.5)
            pygame.quit()


def main():
    """Main entry point"""
    print("=" * 70)
    print("INTEGRATED CHESS SYSTEM")
    print("Camera Detection + Pygame GUI")
    print("=" * 70)
    print("\n📋 Setup:")
    print("   1. Ensure Raspberry Pi camera is streaming")
    print("   2. Position camera above chessboard")
    print("   3. Ensure good lighting")
    print("\n⌨️ Controls:")
    print("   SPACE - Capture position/move")
    print("   R     - Retry initial position")
    print("   Q     - Quit")
    print("=" * 70)

    system = IntegratedChessSystem(STREAM_URL)
    system.run()


if __name__ == "__main__":
    main()