from typing import Dict, Optional

import numpy as np


def negamax_alpha_beta_pruned(
        board: np.ndarray,
        player: int,
        alpha: np.float,
        beta: np.float,
        size: int = 3
) -> Dict[str, int]:
    """
    Simple implementation of the negamax (minimax) algorithm for the tic-tac-toe game. Includes an improvement
    of alpha-beta pruning.

    See tests for example usage.

    :param board: current state of the board
    :param player: the player to make a move (can be 1 or -1)
    :param alpha: the minimum score that the maximizing player is assured of
    :param beta: the maximum score that the minimizing player is assured of
    :param size: size of the board
    :return: dict with results for score and move; the score is given from the perspective of the player who is about
    to play (so score == 1 when player == -1 means that player "-1" won)
    """
    winner = get_winner(board)
    if winner:
        return {'score': winner * player, 'move': None}
    elif check_if_board_is_full(board, size):
        return {'score': 0, 'move': None}

    best_score = -np.inf

    for move in range(size**2):
        row = move // size
        col = move % size
        if board[row, col] == 0:
            copied_board = board.copy()
            copied_board[row, col] = player
            result = negamax_alpha_beta_pruned(copied_board, -player, -beta, -alpha)
            score = -result['score']
            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

    return {'score': best_score, 'move': best_move}


def negamax(board: np.ndarray, player: int, size: int = 3) -> Dict[str, int]:
    """
    Simple implementation of the negamax (minimax) algorithm for the tic-tac-toe game.

    See tests for example usage.

    :param board: current state of the board
    :param player: the player to make a move (can be 1 or -1)
    :param size: size of the board
    :return: dict with results for score and move; the score is given from the perspective of the player who is about
    to play (so score == 1 when player == -1 means that player "-1" won)
    """
    winner = get_winner(board)
    if winner:
        return {'score': winner * player, 'move': None}
    elif check_if_board_is_full(board, size):
        return {'score': 0, 'move': None}

    best_score = -np.inf

    for move in range(size**2):
        row = move // size
        col = move % size
        if board[row, col] == 0:
            copied_board = board.copy()
            copied_board[row, col] = player
            result = negamax(copied_board, -player)
            score = -result['score']
            if score > best_score:
                best_score = score
                best_move = (row, col)

    return {'score': best_score, 'move': best_move}


def check_if_board_is_full(board: np.ndarray, size: int) -> bool:
    return np.count_nonzero(board) == size**2


def get_winner(board: np.ndarray) -> Optional[int]:
    for player in [-1, 1]:
        for i in range(3):
            if board[i, :].tolist().count(player) == 3 or board[:, i].tolist().count(player) == 3:
                return player
        if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
            return player
