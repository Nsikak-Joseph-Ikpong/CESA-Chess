"""
Chess Movement Tracker using YOLO
Tracks chess moves by comparing consecutive images of a chessboard
"""

import cv2
import numpy as np
from ultralytics import YOLO
import chess
import chess.svg
from pathlib import Path

# YOLO model path
WEIGHT_PATH = 'bestV13.pt'

class ChessTracker:
    def __init__(self, weight_path=WEIGHT_PATH):
        """Initialize the chess tracker with YOLO model"""
        self.model = YOLO(weight_path)
        self.board = chess.Board()
        self.previous_board_status = None
        self.conf_threshold = 0.7
        
        # Piece name mapping
        self.piece_names = {
            'P': 'pawn', 'N': 'knight', 'B': 'bishop', 
            'R': 'rook', 'Q': 'queen', 'K': 'king',
            'p': 'pawn', 'n': 'knight', 'b': 'bishop', 
            'r': 'rook', 'q': 'queen', 'k': 'king',
        }
    
    def order_detections(self, boxes, classes):
        """
        Order detected boxes into 8x8 grid representing chessboard
        Returns: 2D array with piece classifications (white/black/empty)
        """
        board_status = []
        detections = []
        
        # Calculate center points for each detection
        for idx, box in enumerate(boxes):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            detections.append({
                'box': idx, 
                'x_center': x_center, 
                'y_center': y_center, 
                'class': classes[idx]
            })
        
        # Sort by y_center to get rows (top to bottom)
        detections = sorted(detections, key=lambda d: d['y_center'])
        
        # Divide into 8 rows
        rows = [detections[i * 8:(i + 1) * 8] for i in range(8)]
        
        # Sort each row by x_center (left to right)
        for row in rows:
            sorted_row = sorted(row, key=lambda d: d['x_center'])
            board_status.append([cell['class'] for cell in sorted_row])
        
        return board_status
    
    def map_board_to_status(self, board):
        """Convert chess.Board to board_status format"""
        board_status = [['empty' for _ in range(8)] for _ in range(8)]
        
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                
                if piece:
                    board_status[row][col] = 'white' if piece.color else 'black'
        
        return board_status
    
    def detect_board_from_image(self, image_path):
        """
        Detect chessboard state from image
        Returns: board_status (8x8 array), annotated image, detection count
        """
        # Run YOLO prediction
        results = self.model.predict(source=image_path, conf=self.conf_threshold)
        
        if not results or len(results[0].boxes.xyxy) == 0:
            return None, None, 0
        
        boxes_count = len(results[0].boxes.xyxy)
        
        # Get detection information
        boxes = results[0].boxes.xyxy.cpu().numpy()
        predicted_classes = results[0].boxes.cls
        class_names = self.model.names
        predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]
        
        # Create annotated image
        annotated_image = results[0].plot()
        
        # Only process if exactly 64 boxes detected
        if boxes_count == 64:
            board_status = self.order_detections(boxes, predicted_class_names)
            return board_status, annotated_image, boxes_count
        else:
            return None, annotated_image, boxes_count
    
    def detect_move(self, previous_status, new_status):
        """
        Detect chess move by comparing board states
        Returns: dictionary with move information
        """
        move = {}
        
        for row in range(8):
            for col in range(8):
                if previous_status[row][col] != new_status[row][col]:
                    square = chess.square(col, 7 - row)
                    square_name = chess.square_name(square)
                    piece = self.board.piece_at(square)
                    
                    # Piece moved FROM this square (now empty)
                    if new_status[row][col] == 'empty' and previous_status[row][col] != 'empty':
                        move['start'] = square_name
                        move['piece'] = self.piece_names.get(piece.symbol(), '') if piece else ''
                    
                    # Piece moved TO this square (was empty or captured)
                    elif previous_status[row][col] == 'empty' and new_status[row][col] != 'empty':
                        move['end'] = square_name
                    
                    # Piece captured (color changed)
                    elif previous_status[row][col] != new_status[row][col]:
                        move['end'] = square_name
                        move['eliminated'] = self.piece_names.get(piece.symbol(), '') if piece else ''
        
        # Check for castling
        if 'start' in move and 'end' in move:
            if move['start'] == "e1" and move['end'] == "g1":
                move['castle'] = "white_kingside"
            elif move['start'] == "e1" and move['end'] == "c1":
                move['castle'] = "white_queenside"
            elif move['start'] == "e8" and move['end'] == "g8":
                move['castle'] = "black_kingside"
            elif move['start'] == "e8" and move['end'] == "c8":
                move['castle'] = "black_queenside"
        
        return move
    
    def is_legal_move(self, move):
        """Check if detected move is legal"""
        if 'start' not in move or 'end' not in move:
            return False, "Incomplete move detected"
        
        try:
            chess_move = chess.Move.from_uci(f"{move['start']}{move['end']}")
            
            if chess_move in self.board.legal_moves:
                return True, "Legal move"
            else:
                return False, "Illegal move - violates chess rules"
        except:
            return False, "Invalid move format"
    
    def apply_move(self, move):
        """Apply move to the board if legal"""
        is_legal, message = self.is_legal_move(move)
        
        if is_legal:
            chess_move = chess.Move.from_uci(f"{move['start']}{move['end']}")
            self.board.push(chess_move)
            self.previous_board_status = self.map_board_to_status(self.board)
            return True, f"Move applied: {move.get('piece', 'piece')} from {move['start']} to {move['end']}"
        else:
            return False, message
    
    def reset_game(self):
        """Reset the board to starting position"""
        self.board = chess.Board()
        self.previous_board_status = self.map_board_to_status(self.board)
    
    def visualize_board_status(self, board_status):
        """Create a simple visualization of board status"""
        print("\nBoard Status (W=White, B=Black, .=Empty):")
        print("  a b c d e f g h")
        for idx, row in enumerate(board_status):
            row_num = 8 - idx
            row_display = []
            for cell in row:
                if cell == 'white':
                    row_display.append('W')
                elif cell == 'black':
                    row_display.append('B')
                else:
                    row_display.append('.')
            print(f"{row_num} {' '.join(row_display)}")
        print()

    def create_movement_heatmap(self, previous_status, new_status, move_info=None):
        """
        Create a side-by-side heatmap showing board states and detected movement
        Left: Previous state, Right: Movement detection
        Returns: matplotlib figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Color mapping for pieces
        color_map = {
            'white': [0.8, 0.8, 0.8],  # Light gray
            'black': [0.2, 0.2, 0.2],  # Dark gray
            'empty': [1.0, 1.0, 1.0]   # White
        }

        # Create color matrices
        prev_matrix = [[color_map[cell] for cell in row] for row in previous_status]

        # Create movement detection matrix
        movement_matrix = []
        for row in range(8):
            row_colors = []
            for col in range(8):
                if previous_status[row][col] != new_status[row][col]:
                    # Changed square - highlight in color
                    if new_status[row][col] == 'empty':
                        # Piece left this square (source)
                        row_colors.append([0.9, 0.6, 0.6])  # Light red
                    else:
                        # Piece arrived at this square (destination)
                        row_colors.append([0.6, 0.9, 0.6])  # Light green
                else:
                    # No change
                    row_colors.append(color_map[previous_status[row][col]])
            movement_matrix.append(row_colors)

        # Plot previous state
        ax1.imshow(prev_matrix, extent=[0, 8, 0, 8], origin='lower')
        ax1.set_title('Previous Position', fontsize=14, fontweight='bold')
        ax1.set_xlabel('File (a-h)', fontsize=10)
        ax1.set_ylabel('Rank (1-8)', fontsize=10)
        ax1.set_xticks(range(9))
        ax1.set_yticks(range(9))
        ax1.set_xticklabels(['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax1.set_yticklabels(['', '1', '2', '3', '4', '5', '6', '7', '8'])
        ax1.grid(True, which='both', color='black', linewidth=1.5)

        # Plot movement detection
        ax2.imshow(movement_matrix, extent=[0, 8, 0, 8], origin='lower')
        ax2.set_title('Movement Detection', fontsize=14, fontweight='bold')
        ax2.set_xlabel('File (a-h)', fontsize=10)
        ax2.set_ylabel('Rank (1-8)', fontsize=10)
        ax2.set_xticks(range(9))
        ax2.set_yticks(range(9))
        ax2.set_xticklabels(['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax2.set_yticklabels(['', '1', '2', '3', '4', '5', '6', '7', '8'])
        ax2.grid(True, which='both', color='black', linewidth=1.5)

        # Add legend for movement detection
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.9, 0.6, 0.6], label='Source (piece left)'),
            Patch(facecolor=[0.6, 0.9, 0.6], label='Destination (piece arrived)'),
            Patch(facecolor=[0.8, 0.8, 0.8], label='White piece (unchanged)'),
            Patch(facecolor=[0.2, 0.2, 0.2], label='Black piece (unchanged)'),
            Patch(facecolor=[1.0, 1.0, 1.0], label='Empty square')
        ]
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

        # Add move annotation if available
        if move_info and 'start' in move_info and 'end' in move_info:
            move_text = f"Move: {move_info.get('piece', 'Piece')} {move_info['start']} → {move_info['end']}"
            if 'eliminated' in move_info:
                move_text += f"\nCaptured: {move_info['eliminated']}"
            fig.suptitle(move_text, fontsize=12, fontweight='bold', y=0.98)

        plt.tight_layout()
        return fig


def main():
    """Example usage of ChessTracker"""

    # Initialize tracker
    tracker = ChessTracker()

    print("=" * 60)
    print("Chess Movement Tracker - Image Mode")
    print("=" * 60)

    # Example: Track moves from two consecutive images
    image1_path = "chess_position1.jpg"  # First position
    image2_path = "chess_position2.jpg"  # After move

    print(f"\n1. Processing first image: {image1_path}")
    board_status1, annotated1, count1 = tracker.detect_board_from_image(image1_path)

    if board_status1 is None:
        print(f"   ❌ Error: Detected {count1} boxes (expected 64)")
        if annotated1 is not None:
            cv2.imwrite("detection1_debug.jpg", annotated1)
            print(f"   Saved debug image to: detection1_debug.jpg")
        return

    print(f"   ✓ Successfully detected {count1} boxes")
    tracker.previous_board_status = board_status1
    tracker.visualize_board_status(board_status1)

    # Save annotated image
    cv2.imwrite("detection1.jpg", annotated1)
    print(f"   Saved annotated image to: detection1.jpg")

    print(f"\n2. Processing second image: {image2_path}")
    board_status2, annotated2, count2 = tracker.detect_board_from_image(image2_path)

    if board_status2 is None:
        print(f"   ❌ Error: Detected {count2} boxes (expected 64)")
        if annotated2 is not None:
            cv2.imwrite("detection2_debug.jpg", annotated2)
            print(f"   Saved debug image to: detection2_debug.jpg")
        return

    print(f"   ✓ Successfully detected {count2} boxes")
    tracker.visualize_board_status(board_status2)

    # Save annotated image
    cv2.imwrite("detection2.jpg", annotated2)
    print(f"   Saved annotated image to: detection2.jpg")

    print("\n3. Detecting move...")
    move = tracker.detect_move(board_status1, board_status2)

    if move:
        print(f"   Move detected:")
        print(f"   - Piece: {move.get('piece', 'Unknown')}")
        print(f"   - From: {move.get('start', 'Unknown')}")
        print(f"   - To: {move.get('end', 'Unknown')}")
        if 'eliminated' in move:
            print(f"   - Captured: {move['eliminated']}")
        if 'castle' in move:
            print(f"   - Castle: {move['castle']}")

        # Validate and apply move
        print("\n4. Validating move...")
        success, message = tracker.apply_move(move)

        if success:
            print(f"   ✓ {message}")
            print(f"\n   Current board FEN: {tracker.board.fen()}")
        else:
            print(f"   ❌ {message}")
    else:
        print("   No move detected (boards are identical)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Check if weight file exists
    if not Path(WEIGHT_PATH).exists():
        print(f"Error: Weight file not found at {WEIGHT_PATH}")
        print("Please ensure the YOLO weight file is in the correct location.")
    else:
        main()