from typing import Dict, Optional

import numpy as np


def minimax(board: np.ndarray, player: int, depth_level: int = 0, size: int = 3) -> Dict[str, int]:
    if get_winner(board):
        return {'score': get_winner(board), 'move': None, 'depth_level': depth_level}
    elif check_if_board_is_full(board):
        return {'score': 0, 'move': None, 'depth_level': depth_level}

    scores = []
    moves = []
    depths = []
    depth_level += 1

    for row in range(size):
        for col in range(size):
            if board[row, col] == 0:
                copied_board = board.copy()
                copied_board[row, col] = player
                result = minimax(copied_board, -player, depth_level)
                scores.append(result['score'])
                depths.append(result['depth_level'])
                moves.append((row, col))

    if player == 1:
        scores_index = scores.index(max(scores))
    else:
        scores_index = scores.index(min(scores))
    return {'score': scores[scores_index], 'move': moves[scores_index], 'depth_level': depths[scores_index]}


def check_if_board_is_full(board: np.ndarray) -> bool:
    return np.count_nonzero(board) == 9


def get_winner(board: np.ndarray) -> Optional[int]:
    for player in [-1, 1]:
        for i in range(3):
            if board[i, :].tolist().count(player) == 3 or board[:, i].tolist().count(player) == 3:
                return player
        if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
            return player
