import numpy as np
import pytest

from scripts.utils import check_if_board_is_full, get_winner, negamax, negamax_alpha_beta_pruned

board0 = np.zeros(shape=(3, 3))
board1 = np.array([[-1, 0, 1], [1, 0, 0], [1, -1, -1]])
board2 = np.array([[1, 0, 1], [0, 0, 0], [0, -1, -1]])
board3 = np.array([[1, -1, -1], [-1, 1, 1], [1, -1, -1]])
board4 = np.array([[1, 0, 0], [0, 0, -1], [0, 0, 0]])
board5 = np.array([[1, 1, -1], [0, 0, -1], [0, 0, 0]])
board6 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
"""
board0:
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])

board1:
array([[-1,  0,  1],
       [ 1,  0,  0],
       [ 1, -1, -1]])

board2:
array([[ 1,  0,  1],
       [ 0,  0,  0],
       [ 0, -1, -1]])

board3:
array([[ 1, -1, -1],
       [-1,  1,  1],
       [ 1, -1, -1]])

board4:
array([[ 1,  0,  0],
       [ 0,  0, -1],
       [ 0,  0,  0]])

board5:
array([[ 1,  1, -1],
       [ 0,  0, -1],
       [ 0,  0,  0]])

board6:
array([[ 0,  0,  0],
       [ 0,  1,  0],
       [ 0,  0,  0]])
"""


@pytest.mark.parametrize("board, expected", [
    (board0, False),
    (board1, False),
    (board2, False),
    (board3, True),
])
def test_check_if_board_is_full(board, expected):
    assert check_if_board_is_full(board, 3) == expected


@pytest.mark.parametrize("board, expected", [
    (np.array([[-1, 0, 1], [1, -1, 0], [1, -1, -1]]), -1),
    (np.array([[-1, 0, 1], [1, 1, 0], [1, -1, -1]]), 1),
])
def test_get_winner_when_game_is_decided(board, expected):
    assert get_winner(board) == expected


@pytest.mark.parametrize("board, expected", [
    (board1, None),
    (board3, None),
])
def test_get_winner_when_game_is_not_decided(board, expected):
    assert get_winner(board) is expected


@pytest.mark.parametrize("board, player, expected", [
    (board0, 1, 0),
    (board0, -1, 0),
    (board6, -1, 0),
    (board1, 1, 1),
    (board1, -1, 1),
    (board2, 1, 1),
    (board2, -1, 1),
    (board4, 1, 1),
    (board5, 1, -1),
    (board6, 1, 1),
])
def test_negamax_whether_predicts_result(board, player, expected):
    # watch out! the negamax function returns the results from the perspective
    # of the player to play, so that a line `(board1, -1, 1)` expects player "-1" to win
    assert negamax(board, player)['score'] == expected


@pytest.mark.parametrize("board, player, expected", [
    (board0, 1, 0),
    (board0, -1, 0),
    (board6, -1, 0),
    (board1, 1, 1),
    (board1, -1, 1),
    (board2, 1, 1),
    (board2, -1, 1),
    (board4, 1, 1),
    (board5, 1, -1),
])
def test_negamax_alpha_beta_pruned_whether_predicts_result(board, player, expected):
    # watch out! the negamax_alpha_beta_pruned function returns the results from the perspective
    # of the player to play, so that a line `(board1, -1, 1)` expects player "-1" to win
    assert negamax_alpha_beta_pruned(board, player, -np.inf, np.inf)['score'] == expected


@pytest.mark.parametrize("board, player, expected", [
    (board1, 1, [(1, 1)]),
    (board2, 1, [(0, 1)]),
    (board2, -1, [(2, 0), (0, 1)]),
])
def test_negamax_plays_proper_move(board, player, expected):
    assert negamax(board, player)['move'] in expected


@pytest.mark.parametrize("board, player, expected", [
    (board1, 1, [(1, 1)]),
    (board2, 1, [(0, 1)]),
    (board2, -1, [(2, 0), (0, 1)]),
])
def test_negamax_alpha_beta_pruned_plays_proper_move(board, player, expected):
    assert negamax_alpha_beta_pruned(board, player, -np.inf, np.inf)['move'] in expected
