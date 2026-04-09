import cv2
import matplotlib.pyplot as plt
from chess_tracker import ChessTracker
import numpy as np
import time
import os

# Replace with your Pi's IP
PI_IP = "10.223.89.228"
#PI_IP = "172.20.10.3"
stream_url = f"http://{PI_IP}:5000/video_feed"


class LiveChessTracker:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.tracker = ChessTracker()
        self.cap = None
        self.current_board_status = None
        self.move_count = 0
        self.min_squares_threshold = 60
        self.detection_history = []  # Track recent detections for adaptive thresholds

    def connect(self):
        """Connect to the Pi camera stream with retry logic"""
        print(f"Connecting to {self.stream_url}...")

        # Set buffer size to reduce latency and corrupted frames
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer

        if not self.cap.isOpened():
            print("Error: Could not connect to stream")
            return False

        print("✓ Connected to Pi camera stream")
        return True

    def reconnect_if_needed(self):
        """Reconnect to stream if it becomes unstable"""
        if self.cap is None or not self.cap.isOpened():
            print("⚠️  Stream disconnected - reconnecting...")
            self.cap.release()
            time.sleep(1)
            return self.connect()
        return True

    def assess_frame_quality(self, frame):
        """Check if frame is suitable for detection"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check brightness
        brightness = np.mean(gray)

        # Check blur (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check contrast
        contrast = gray.std()

        quality_score = {
            'brightness': brightness,  # Should be 40-220
            'sharpness': laplacian_var,  # Should be > 100
            'contrast': contrast,  # Should be > 30
            'acceptable': (40 < brightness < 220 and
                           laplacian_var > 100 and
                           contrast > 30)
        }

        return quality_score

    def suggest_lighting_adjustment(self, frame):
        """Analyze and suggest lighting improvements"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Analyze dark regions (likely black pieces)
        dark_regions = gray < 50
        dark_percentage = (dark_regions.sum() / gray.size) * 100

        # Analyze bright regions
        bright_regions = gray > 200
        bright_percentage = (bright_regions.sum() / gray.size) * 100

        mean_brightness = np.mean(gray)

        if mean_brightness < 80:
            return "⚠️  Too dark overall - add general lighting"
        elif dark_percentage > 40:
            return "⚠️  Too many dark areas - add light above black pieces"
        elif bright_percentage > 30:
            return "⚠️  Too bright - reduce lighting or exposure"
        else:
            return "✓ Lighting looks acceptable"

    def capture_frame(self):
        """Capture a frame with brief stabilization and error handling"""
        # Flush buffer by reading several frames to clear corrupted data
        for _ in range(15):  # Increased to flush more thoroughly
            try:
                self.cap.read()
            except:
                pass  # Ignore errors during buffer flush

        # Longer delay for better stabilization
        time.sleep(0.5)

        # Try to capture a valid frame with retries
        for attempt in range(3):
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    return frame
            except Exception as e:
                # MJPEG stream error - retry
                time.sleep(0.1)
                continue

        return None

    def create_empty_board_status(self):
        """Create an empty 8x8 board status structure"""
        return {row: {col: None for col in range(8)} for row in range(8)}

    def count_pieces_in_status(self, status):
        """Count black and white pieces in board status (handles both dict and list)"""
        black_count = 0
        white_count = 0

        if status is None:
            return 0, 0

        # Handle dict format
        if isinstance(status, dict):
            for row in range(8):
                for col in range(8):
                    piece = status.get(row, {}).get(col, None)
                    if piece:
                        piece_str = str(piece).lower()
                        if 'black' in piece_str:
                            black_count += 1
                        elif 'white' in piece_str:
                            white_count += 1
        # Handle list format
        elif isinstance(status, list):
            for row in status:
                if isinstance(row, list):
                    for piece in row:
                        if piece:
                            piece_str = str(piece).lower()
                            if 'black' in piece_str:
                                black_count += 1
                            elif 'white' in piece_str:
                                white_count += 1

        return black_count, white_count

    def verify_detection_consistency(self, status1, status2):
        """Check if two consecutive detections are similar"""
        if status1 is None or status2 is None:
            return False

        count1_b, count1_w = self.count_pieces_in_status(status1)
        count2_b, count2_w = self.count_pieces_in_status(status2)

        # Piece counts shouldn't change wildly between frames
        diff_b = abs(count1_b - count2_b)
        diff_w = abs(count1_w - count2_w)

        return diff_b <= 2 and diff_w <= 2  # Allow small variation

    def update_adaptive_threshold(self, recent_count):
        """Adjust threshold based on recent performance"""
        self.detection_history.append(recent_count)
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)

        # If consistently getting 62-64, we can trust that range
        if len(self.detection_history) >= 3:
            avg_recent = np.mean(self.detection_history)
            if avg_recent >= 62:
                self.min_squares_threshold = 60
            elif avg_recent >= 58:
                self.min_squares_threshold = 55
            else:
                self.min_squares_threshold = 60  # Keep default

    def diagnose_piece_detection(self, board_before, board_after):
        """Show detailed comparison of what changed between two board states"""
        print("\n🔬 PIECE DETECTION DIAGNOSIS:")
        print("=" * 50)

        # Count pieces
        before_blacks, before_whites = self.count_pieces_in_status(board_before)
        after_blacks, after_whites = self.count_pieces_in_status(board_after)

        before_empty = 64 - before_blacks - before_whites
        after_empty = 64 - after_blacks - after_whites

        changes = []

        # Try to detect changes (handle both formats)
        try:
            if isinstance(board_before, dict) and isinstance(board_after, dict):
                for row in range(8):
                    for col in range(8):
                        square_name = chr(ord('a') + col) + str(8 - row)
                        before_piece = board_before.get(row, {}).get(col, None)
                        after_piece = board_after.get(row, {}).get(col, None)

                        if before_piece != after_piece:
                            changes.append({
                                'square': square_name,
                                'before': before_piece,
                                'after': after_piece
                            })
                            print(f"   {square_name}: {before_piece} → {after_piece}")
            elif isinstance(board_before, list) and isinstance(board_after, list):
                for row in range(min(len(board_before), len(board_after))):
                    for col in range(min(len(board_before[row]), len(board_after[row]))):
                        square_name = chr(ord('a') + col) + str(8 - row)
                        before_piece = board_before[row][col]
                        after_piece = board_after[row][col]

                        if before_piece != after_piece:
                            changes.append({
                                'square': square_name,
                                'before': before_piece,
                                'after': after_piece
                            })
                            print(f"   {square_name}: {before_piece} → {after_piece}")
        except Exception as e:
            print(f"   Could not detect individual changes: {e}")

        print(f"\nPiece counts:")
        print(f"   BEFORE - Black: {before_blacks}, White: {before_whites}, Empty: {before_empty}")
        print(f"   AFTER  - Black: {after_blacks}, White: {after_whites}, Empty: {after_empty}")

        if not changes:
            print("\n⚠️  NO CHANGES DETECTED!")
            print("   This suggests:")
            if after_blacks < before_blacks:
                print(f"   • {before_blacks - after_blacks} BLACK PIECES DISAPPEARED")
                print("   • Add MORE LIGHT for black pieces!")
            if after_whites < before_whites:
                print(f"   • {before_whites - after_whites} white pieces disappeared")
            if after_blacks == before_blacks and after_whites == before_whites:
                print("   • Piece counts unchanged")
                print("   • Board state may be identical")

        print("=" * 50)
        return changes

    def debug_status_object(self, status, label="Status"):
        """Debug helper to inspect status object structure"""
        print(f"\n🔍 DEBUG - {label}:")
        print(f"   Type: {type(status)}")

        if status is None:
            print(f"   Value: None")
            return

        if isinstance(status, dict):
            print(f"   Keys: {list(status.keys())[:5]}...")  # First 5 keys
            if 0 in status:
                print(f"   Row 0 type: {type(status[0])}")
                print(f"   Row 0 content: {status[0]}")
            if len(status) > 0:
                first_key = list(status.keys())[0]
                print(f"   First row ({first_key}): {status[first_key]}")
        elif isinstance(status, list):
            print(f"   Length: {len(status)}")
            if len(status) > 0:
                print(f"   First row type: {type(status[0])}")
                print(f"   First row: {status[0]}")
        else:
            print(f"   Value: {status}")

    def capture_board_with_retries(self, max_attempts=6):
        """Enhanced capture with quality filtering and better candidate selection"""
        candidates = []

        print(f"\n📷 Attempting to capture board (up to {max_attempts} tries)...")

        for attempt in range(max_attempts):
            # Capture frame
            frame = self.capture_frame()
            if frame is None:
                print(f"   Attempt {attempt + 1}/{max_attempts}: Failed to capture frame")
                continue

            # CHECK QUALITY FIRST
            quality = self.assess_frame_quality(frame)

            if not quality['acceptable']:
                print(
                    f"   Attempt {attempt + 1}/{max_attempts}: Poor quality (brightness={quality['brightness']:.1f}, sharpness={quality['sharpness']:.1f})")
                lighting_suggestion = self.suggest_lighting_adjustment(frame)
                print(f"      {lighting_suggestion}")
                time.sleep(0.5)
                continue

            # Only process good-quality frames
            temp_path = f"temp_attempt_{attempt}.jpg"
            cv2.imwrite(temp_path, frame)

            # Get detection result
            status, annotated, count = self.tracker.detect_board_from_image(temp_path)

            # 🔍 DEBUG - Check what we got back
            if attempt == 0:  # Only debug first attempt to avoid spam
                self.debug_status_object(status, f"Attempt {attempt + 1} Status")

            # Count black pieces detected
            black_count, white_count = self.count_pieces_in_status(status)

            # 🔍 Additional debug info
            print(
                f"   Attempt {attempt + 1}/{max_attempts}: {count} squares ({black_count}B, {white_count}W) Sharpness={quality['sharpness']:.0f}")

            # If we got 0 pieces but YOLO detected pieces, there's a parsing problem
            if black_count == 0 and white_count == 0 and count > 0:
                print(f"   ⚠️  WARNING: YOLO detected squares but piece count is 0!")
                print(f"   ⚠️  This indicates a data structure mismatch in ChessTracker")

            candidates.append({
                'count': count,
                'black': black_count,
                'white': white_count,
                'frame': frame,
                'status': status,
                'annotated': annotated,
                'quality': quality,
                'attempt': attempt
            })

            # If perfect, use it immediately
            if count == 64 and status is not None and black_count > 0:  # Also check pieces exist
                print(f"   ✓ Perfect detection!")
                # Clean up temp files
                for i in range(attempt + 1):
                    try:
                        os.remove(f"temp_attempt_{i}.jpg")
                    except:
                        pass
                return status, annotated, count, frame

            time.sleep(0.5)

        # Clean up temp files
        for i in range(max_attempts):
            try:
                os.remove(f"temp_attempt_{i}.jpg")
            except:
                pass

        # Pick best candidate
        if candidates:
            best = max(candidates, key=lambda x: (x['count'], x['black'] + x['white'], x['quality']['sharpness']))

            print(f"\n   Best result: {best['count']} squares, {best['black']} black, {best['white']} white")

            if best['count'] >= self.min_squares_threshold:
                print(f"   ✓ Acceptable detection ({best['count']}/64 squares)")
                return best['status'], best['annotated'], best['count'], best['frame']
            else:
                print(f"   ❌ Best result below threshold")
                return None, None, 0, None
        else:
            print(f"\n   ❌ No valid candidates")
            return None, None, 0, None

    def process_initial_position(self):
        """Capture and process the initial board position"""
        print("\n" + "=" * 50)
        print("INITIAL BOARD DETECTION")
        print("=" * 50)
        print("\n💡 Setup Tips:")
        print("   • Position camera directly above board")
        print("   • Ensure GOOD LIGHTING (especially for black pieces)")
        print("   • Entire board must be visible")
        print("   • Minimize shadows on black pieces")
        print(f"\n   Minimum required: {self.min_squares_threshold}/64 squares")

        input("\nPress ENTER when ready to capture...")

        # Show countdown with lighting check
        for i in range(3, 0, -1):
            ret, frame = self.cap.read()
            if ret:
                display = frame.copy()

                lighting_msg = self.suggest_lighting_adjustment(frame)
                quality = self.assess_frame_quality(frame)

                cv2.putText(display, f"Capturing in {i}...", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                color = (0, 255, 0) if quality['acceptable'] else (0, 165, 255)
                cv2.putText(display, lighting_msg, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(display, f"Brightness: {quality['brightness']:.1f}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display, f"Sharpness: {quality['sharpness']:.1f}", (10, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Live Chess Tracker', display)
                cv2.waitKey(1)
            time.sleep(1)

        # Try multiple captures
        status, annotated, count, frame = self.capture_board_with_retries(max_attempts=6)

        if status is None or count == 0:
            print(f"\n❌ Failed to capture board")
            print("\n💡 Troubleshooting:")
            print("   1. Add more light")
            print("   2. Adjust camera angle")
            print("   3. Ensure board is fully visible")
            return False

        if count < self.min_squares_threshold:
            print(f"\n❌ Insufficient squares detected. Found {count}/{self.min_squares_threshold} minimum")
            if annotated is not None:
                cv2.imwrite("failed_detection.jpg", annotated)
                cv2.imshow('Failed Detection', annotated)
                cv2.waitKey(5000)
            return False

        black_count, white_count = self.count_pieces_in_status(status)

        if count < 64:
            print(f"\n⚠️  WARNING: Partial board detected ({count}/64 squares)")
        else:
            print(f"\n✓ Perfect detection! ({count} squares)")

        print(f"   Detected pieces: {black_count} black, {white_count} white")

        self.update_adaptive_threshold(count)
        self.current_board_status = status
        self.tracker.previous_board_status = status

        cv2.imwrite("initial_board.jpg", annotated)
        print("   Saved: initial_board.jpg")

        cv2.imshow('Initial Position', annotated)
        cv2.waitKey(2000)

        return True

    def wait_for_move(self):
        """Wait for player to make a move"""
        print("\n" + "=" * 50)
        print(f"WAITING FOR MOVE #{self.move_count + 1}")
        print("=" * 50)

        waiting = True
        while waiting:
            ret, frame = self.cap.read()
            if not ret:
                return None

            display = frame.copy()
            quality = self.assess_frame_quality(frame)
            quality_color = (0, 255, 0) if quality['acceptable'] else (0, 165, 255)

            cv2.putText(display, f"Move #{self.move_count + 1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display, "Make your move", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Press SPACE when done", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Press 'r' to retry initial", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Press 'q' to quit", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Light: {'GOOD' if quality['acceptable'] else 'POOR'}",
                        (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

            cv2.imshow('Live Chess Tracker', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                waiting = False
            elif key == ord('r'):
                return 'retry'
            elif key == ord('q'):
                return None

        # Countdown
        for i in range(3, 0, -1):
            ret, frame = self.cap.read()
            if ret:
                display = frame.copy()
                cv2.putText(display, f"Capturing in {i}...", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('Live Chess Tracker', display)
                cv2.waitKey(1)
            time.sleep(1)

        status, annotated, count, frame = self.capture_board_with_retries(max_attempts=6)

        return {'status': status, 'annotated': annotated, 'count': count, 'frame': frame}

    def track_live_game(self):
        """Main loop to track a live chess game"""
        if not self.connect():
            return

        if not self.process_initial_position():
            print("\n❌ Failed to initialize. Exiting.")
            self.cap.release()
            cv2.destroyAllWindows()
            return

        print("\n" + "=" * 50)
        print("LIVE CHESS TRACKING STARTED!")
        print("=" * 50)
        print(f"Accepting boards with {self.min_squares_threshold}+ squares")
        print("=" * 50)

        try:
            while True:
                result = self.wait_for_move()

                if result is None:
                    print("\n👋 Exiting...")
                    break

                if result == 'retry':
                    print("\n🔄 Retrying initial position...")
                    if not self.process_initial_position():
                        continue
                    else:
                        continue

                status_new = result['status']
                annotated = result['annotated']
                count = result['count']
                frame = result['frame']

                if status_new is None or count == 0:
                    print(f"\n❌ Failed to capture board - Auto-retrying...")
                    # Automatically retry up to 4 more times
                    retry_success = False
                    for retry in range(1, 5):
                        print(f"   Auto-retry {retry}/4...")
                        time.sleep(1)
                        retry_result = self.wait_for_move()

                        if retry_result is None or retry_result == 'retry':
                            continue

                        if retry_result['status'] is not None and retry_result['count'] > 0:
                            print(f"   ✓ Retry successful!")
                            result = retry_result
                            status_new = result['status']
                            annotated = result['annotated']
                            count = result['count']
                            frame = result['frame']
                            retry_success = True
                            break

                    if not retry_success:
                        print(f"\n❌ All retries failed. Press SPACE to try manually or 'q' to quit")
                        while True:
                            key = cv2.waitKey(0) & 0xFF
                            if key == ord(' '):
                                break
                            elif key == ord('q'):
                                self.cap.release()
                                cv2.destroyAllWindows()
                                return
                        continue
                    # If retry was successful, continue with the new result

                if count < self.min_squares_threshold:
                    print(f"\n❌ Insufficient squares: {count}/{self.min_squares_threshold}")
                    print("   Press SPACE to retry or 'q' to quit")
                    if annotated is not None:
                        cv2.imshow('Failed Detection', annotated)
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return
                    continue

                self.update_adaptive_threshold(count)

                if count < 64:
                    print(f"   ⚠️  Partial detection: {count}/64 squares")
                else:
                    print(f"   ✓ Perfect detection: {count}/64 squares")

                if self.current_board_status is None:
                    print("   ⚠️  No previous board status")
                    self.current_board_status = status_new
                    continue

                # CRITICAL: Check for color detection flip
                if not self.verify_detection_consistency(self.current_board_status, status_new):
                    print("   🚨 WARNING: Piece count changed by >1!")
                    print("   🚨 This indicates color detection flip - REJECTING")
                    print("   Press SPACE to retry this move")

                    if annotated is not None:
                        cv2.imshow('Color Flip Detected', annotated)

                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return
                    continue

                print("\n🔍 Analyzing move...")
                changes = self.diagnose_piece_detection(self.current_board_status, status_new)

                # Check for too many changes (color flip indicator)
                if len(changes) > 4:
                    print(f"\n🚨 TOO MANY CHANGES ({len(changes)}) - Color flip detected!")
                    print("   Press SPACE to retry")

                    if annotated is not None:
                        cv2.imshow('Too Many Changes', annotated)

                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return
                    continue

                try:
                    move = self.tracker.detect_move(self.current_board_status, status_new)
                except Exception as e:
                    print(f"   ❌ Error detecting move: {e}")
                    print(f"   Press SPACE to continue or 'q' to quit")
                    if annotated is not None:
                        cv2.imshow('Detection Error', annotated)
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return
                    continue

                if 'start' in move and 'end' in move:
                    self.move_count += 1
                    print(f"\n{'=' * 50}")
                    print(f"MOVE #{self.move_count} DETECTED")
                    print(f"{'=' * 50}")
                    print(f"📍 {move.get('piece', 'Piece').capitalize()}: {move['start']} → {move['end']}")

                    if 'eliminated' in move:
                        print(f"   🎯 Captured: {move['eliminated']}")

                    try:
                        success, message = self.tracker.apply_move(move)

                        if success:
                            print(f"✓ {message}")
                            validation_status = "VALID"
                            text_color = (0, 255, 0)
                        else:
                            print(f"⚠️  {message}")
                            print(f"   Accepting move anyway")
                            validation_status = "ACCEPTED"
                            text_color = (0, 165, 255)
                            self.tracker.previous_board_status = status_new
                    except Exception as e:
                        print(f"⚠️  Validation error: {e}")
                        print(f"   Accepting move anyway")
                        validation_status = "ACCEPTED"
                        text_color = (0, 165, 255)
                        self.tracker.previous_board_status = status_new

                    cv2.imwrite(f"move_{self.move_count}.jpg", annotated)

                    try:
                        fig = self.tracker.create_movement_heatmap(
                            self.current_board_status, status_new, move
                        )
                        plt.savefig(f"heatmap_move_{self.move_count}.png",
                                    dpi=200, bbox_inches='tight')
                        plt.close(fig)
                        print(f"💾 Saved: move_{self.move_count}.jpg, heatmap_move_{self.move_count}.png")
                    except:
                        print(f"💾 Saved: move_{self.move_count}.jpg")

                    self.current_board_status = status_new

                    result_display = annotated.copy()
                    cv2.putText(result_display, f"Move {self.move_count}: {move['start']} -> {move['end']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    cv2.putText(result_display, validation_status,
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    if count < 64:
                        cv2.putText(result_display, f"Partial: {count}/64",
                                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('Move Detected', result_display)
                    cv2.waitKey(2000)

                else:
                    print("\n⚠️  NO MOVE DETECTED")
                    if not changes:
                        print("   • Board states are identical")
                    print("\n   Press SPACE to continue, 'r' to retry, 's' to skip")

                    if annotated is not None:
                        cv2.imshow('No Move Detected', annotated)

                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('r'):
                            break
                        elif key == ord('s'):
                            print("   Forcing board state update...")
                            self.current_board_status = status_new
                            self.tracker.previous_board_status = status_new
                            break

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            print(f"\n{'=' * 50}")
            print(f"GAME SUMMARY")
            print(f"{'=' * 50}")
            print(f"  Total moves tracked: {self.move_count}")
            try:
                print(f"  Final FEN: {self.tracker.board.fen()}")
            except:
                print(f"  Final FEN: Unable to generate")

            if self.detection_history:
                avg_detection = np.mean(self.detection_history)
                print(f"  Average detection: {avg_detection:.1f}/64 squares")

            print(f"{'=' * 50}")

            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    print("=" * 50)
    print("LIVE CHESS TRACKER - COLOR FLIP FIX")
    print("=" * 50)
    print("\n📋 Instructions:")
    print("   1. Position camera directly above chessboard")
    print("   2. ADD MORE LIGHT (critical for detection!)")
    print("   3. Make sure board is fully visible")
    print("   4. Follow on-screen prompts")
    print("\n⚠️  IMPORTANT:")
    print("   • System now rejects color detection flips")
    print("   • If >4 squares change, it will ask you to retry")
    print("   • This prevents fake 'captured queen' messages")
    print("\n⌨️  Controls:")
    print("   SPACE - Capture move / Continue")
    print("   S     - Skip & force update board state")
    print("   R     - Retry")
    print("   Q     - Quit")
    print("=" * 50)

    tracker = LiveChessTracker(stream_url)
    tracker.track_live_game()


if __name__ == "__main__":
    main()