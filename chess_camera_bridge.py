"""
Chess Camera Bridge - Enhanced
Extends the original Chess class to accept moves from both camera detection AND manual mouse input
"""

import pygame
import time
from chess_gui import Chess

class ChessCameraBridge(Chess):
    """Extended Chess class that can accept camera-detected moves AND manual mouse moves"""

    def __init__(self, screen, pieces_src, square_coords, square_length):
        super().__init__(screen, pieces_src, square_coords, square_length)
        self.camera_mode = True
        self.pending_camera_move = None
        self.last_move_animation = None

        # Manual move state
        self.selected_piece = None
        self.selected_pos = None
        self.last_click_time = 0  # Track last click time for debouncing

        # Store initialization parameters for reset
        self._init_screen = screen
        self._init_pieces_src = pieces_src
        self._init_square_coords = square_coords
        self._init_square_length = square_length

        # Override turn to neutral state - first move determines who goes first
        self.turn = {"black": 0, "white": 0}

    def reset_game(self):
        """Reset the chess game to starting position"""
        print("♻️  Resetting chess board...")

        # Call parent class __init__ to reset the board
        super().__init__(
            self._init_screen,
            self._init_pieces_src,
            self._init_square_coords,
            self._init_square_length
        )

        # Reset camera bridge specific state
        self.camera_mode = True
        self.pending_camera_move = None
        self.last_move_animation = None

        # Reset manual move state
        self.selected_piece = None
        self.selected_pos = None
        self.last_click_time = 0

        # Reset turn to neutral
        self.turn = {"black": 0, "white": 0}

        print("✓ Chess board reset complete")

    def apply_camera_move(self, move_dict):
        """
        Apply a move detected by the camera

        Args:
            move_dict: Dictionary with 'start', 'end', 'piece', and optionally 'eliminated'
                      Example: {'piece': 'pawn', 'start': 'e2', 'end': 'e4'}

        Returns:
            bool: True if move was applied successfully
        """
        try:
            # Extract move information
            start_square = move_dict.get('start', '')  # e.g., 'e2'
            end_square = move_dict.get('end', '')      # e.g., 'e4'

            if not start_square or not end_square:
                print(f"❌ Invalid move: missing start or end square")
                return False

            # Convert chess notation to board coordinates
            start_col = start_square[0]  # 'e'
            start_row = int(start_square[1])  # 2

            end_col = end_square[0]  # 'e'
            end_row = int(end_square[1])  # 4

            # Convert to array indices (x, y)
            start_x = ord(start_col) - ord('a')
            start_y = 8 - start_row

            end_x = ord(end_col) - ord('a')
            end_y = 8 - end_row

            # Get the piece at start position
            piece_at_start = self.piece_location[start_col][start_row][0]

            if not piece_at_start:
                print(f"❌ No piece at {start_square}")
                return False

            # Check piece color
            piece_color = piece_at_start[:5]  # 'white' or 'black'

            # SPECIAL CASE: First move of the game
            is_first_move = (self.turn['white'] == 0 and self.turn['black'] == 0)

            if is_first_move:
                print(f"ℹ️  First move detected - setting turn to {piece_color}")
                if piece_color == 'white':
                    self.turn['white'] = 1
                    self.turn['black'] = 0
                else:
                    self.turn['black'] = 1
                    self.turn['white'] = 0
            else:
                # Validate turn
                if piece_color == 'white' and self.turn['white'] == 0:
                    print(f"❌ Not white's turn")
                    return False
                elif piece_color == 'black' and self.turn['black'] == 0:
                    print(f"❌ Not black's turn")
                    return False

            # Check if there's a piece to capture at destination
            piece_at_end = self.piece_location[end_col][end_row][0]

            # Store animation data
            self.last_move_animation = {
                'start': [start_x, start_y],
                'end': [end_x, end_y],
                'piece': piece_at_start
            }

            # If capturing
            if piece_at_end:
                # Check if it's opponent's piece
                end_piece_color = piece_at_end[:5]
                if end_piece_color == piece_color:
                    print(f"❌ Cannot capture your own piece")
                    return False

                # Check for king capture (game over)
                if piece_at_end == "white_king":
                    self.winner = "Black"
                    print("🏆 Black wins!")
                elif piece_at_end == "black_king":
                    self.winner = "White"
                    print("🏆 White wins!")

                # Add to captured pieces
                self.captured.append(self.piece_location[end_col][end_row])
                print(f"🎯 Captured: {piece_at_end} at {end_square}")

            # Move the piece
            self.piece_location[end_col][end_row][0] = piece_at_start
            self.piece_location[start_col][start_row][0] = ""

            # Switch turns
            if self.turn["black"]:
                self.turn["black"] = 0
                self.turn["white"] = 1
            else:
                self.turn["black"] = 1
                self.turn["white"] = 0

            print(f"✅ Move applied: {piece_at_start} from {start_square} to {end_square}")
            return True

        except Exception as e:
            print(f"❌ Error applying camera move: {e}")
            import traceback
            traceback.print_exc()
            return False

    def move_piece_manual(self, board_x, board_y):
        """
        Handle manual mouse-based piece movement

        Args:
            board_x: Column index (0-7)
            board_y: Row index (0-7)
        """
        # Debounce: Prevent same click from being processed twice
        current_time = time.time()
        if current_time - self.last_click_time < 0.3:  # 300ms debounce
            return
        self.last_click_time = current_time

        # Convert board coordinates to chess notation
        col_char = chr(ord('a') + board_x)
        row_num = 8 - board_y

        piece_at_pos = self.piece_location[col_char][row_num][0]

        # First click - select a piece
        if self.selected_piece is None:
            if not piece_at_pos:
                print("⚠️  No piece at this square")
                return

            piece_color = piece_at_pos[:5]

            # Check if it's this color's turn
            is_first_move = (self.turn['white'] == 0 and self.turn['black'] == 0)

            if not is_first_move:
                if piece_color == 'white' and self.turn['white'] == 0:
                    print("⚠️  Not white's turn")
                    return
                elif piece_color == 'black' and self.turn['black'] == 0:
                    print("⚠️  Not black's turn")
                    return

            # Select the piece
            self.selected_piece = piece_at_pos
            self.selected_pos = (col_char, row_num, board_x, board_y)

            # Highlight the selected piece
            self.piece_location[col_char][row_num][1] = True

            # Calculate and store possible moves for this piece
            self.moves = self.possible_moves(piece_at_pos, [board_x, board_y])

            print(f"✓ Selected: {piece_at_pos} at {col_char}{row_num}")
            print(f"  Showing {len(self.moves)} possible moves")

        # Second click - move the piece
        else:
            start_col, start_row, start_x, start_y = self.selected_pos
            end_col = col_char
            end_row = row_num

            # Deselect the piece
            self.piece_location[start_col][start_row][1] = False

            # Clear possible moves
            self.moves = []

            # Check if clicking same square (deselect)
            if start_col == end_col and start_row == end_row:
                print("⚠️  Move cancelled")
                self.selected_piece = None
                self.selected_pos = None
                return

            # Check if destination has own piece
            piece_at_dest = self.piece_location[end_col][end_row][0]
            if piece_at_dest:
                dest_color = piece_at_dest[:5]
                source_color = self.selected_piece[:5]

                if dest_color == source_color:
                    # Clicking another piece of same color - switch selection
                    self.selected_piece = piece_at_dest
                    self.selected_pos = (end_col, end_row, board_x, board_y)
                    self.piece_location[end_col][end_row][1] = True

                    # Calculate possible moves for new selection
                    self.moves = self.possible_moves(piece_at_dest, [board_x, board_y])

                    print(f"✓ Selected: {piece_at_dest} at {end_col}{end_row}")
                    print(f"  Showing {len(self.moves)} possible moves")
                    return

            # Apply the move
            move_dict = {
                'start': f"{start_col}{start_row}",
                'end': f"{end_col}{end_row}",
                'piece': self.selected_piece.split('_')[1] if '_' in self.selected_piece else 'piece'
            }

            if piece_at_dest:
                move_dict['eliminated'] = piece_at_dest.split('_')[1] if '_' in piece_at_dest else 'piece'

            success = self.apply_camera_move(move_dict)

            # Reset selection
            self.selected_piece = None
            self.selected_pos = None

    def draw_last_move_highlight(self):
        """Draw a highlight showing the last move made"""
        if self.last_move_animation:
            # Yellow transparent overlay for last move
            highlight_color = (255, 255, 0, 100)
            surface = pygame.Surface((self.square_length, self.square_length), pygame.SRCALPHA)
            surface.fill(highlight_color)

            start_x, start_y = self.last_move_animation['start']
            end_x, end_y = self.last_move_animation['end']

            # Highlight start and end squares
            self.screen.blit(surface, self.board_locations[start_x][start_y])
            self.screen.blit(surface, self.board_locations[end_x][end_y])

    def draw_pieces(self):
        """Override to add move highlighting"""
        # Draw last move highlight first (so it's behind pieces)
        self.draw_last_move_highlight()

        # Draw selection highlight and possible moves for manual mode
        transparent_green = (0, 194, 39, 170)
        transparent_blue = (28, 21, 212, 170)

        surface_green = pygame.Surface((self.square_length, self.square_length), pygame.SRCALPHA)
        surface_green.fill(transparent_green)

        surface_blue = pygame.Surface((self.square_length, self.square_length), pygame.SRCALPHA)
        surface_blue.fill(transparent_blue)

        # Highlight selected piece AND possible moves
        for val in self.piece_location.values():
            for value in val.values():
                piece_name = value[0]
                piece_coord_x, piece_coord_y = value[2]

                if value[1] and len(value[0]) > 5:  # Piece is selected
                    if value[0][:5] == "black":
                        # Highlight selected square
                        self.screen.blit(surface_green, self.board_locations[piece_coord_x][piece_coord_y])

                        # Highlight possible moves in same color
                        if len(self.moves) > 0:
                            for move in self.moves:
                                x_coord = move[0]
                                y_coord = move[1]
                                if x_coord >= 0 and y_coord >= 0 and x_coord < 8 and y_coord < 8:
                                    self.screen.blit(surface_green, self.board_locations[x_coord][y_coord])

                    elif value[0][:5] == "white":
                        # Highlight selected square
                        self.screen.blit(surface_blue, self.board_locations[piece_coord_x][piece_coord_y])

                        # Highlight possible moves in same color
                        if len(self.moves) > 0:
                            for move in self.moves:
                                x_coord = move[0]
                                y_coord = move[1]
                                if x_coord >= 0 and y_coord >= 0 and x_coord < 8 and y_coord < 8:
                                    self.screen.blit(surface_blue, self.board_locations[x_coord][y_coord])

        # Draw all chess pieces
        for val in self.piece_location.values():
            for value in val.values():
                piece_name = value[0]
                piece_coord_x, piece_coord_y = value[2]

                if len(value[0]) > 1:
                    self.chess_pieces.draw(self.screen, piece_name,
                                          self.board_locations[piece_coord_x][piece_coord_y])