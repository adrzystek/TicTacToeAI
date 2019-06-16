import numpy as np

from utils import check_if_board_is_full, get_winner, negamax_alpa_beta_pruned

human_player = int(input('Do you want to play as the first [1] or as the second [-1] player?\n'))
computer_player = 1 if human_player == -1 else -1

size = 3

board = np.zeros(shape=(size, size))

print(f'\nInitial state of the board:\n{board}\n')

if human_player == 1:
    while True:
        move = input('Play a move. (type e.g. "0, 0" for a move in the left upper corner)\n')

        try:
            move = [int(m) for m in move.split(',')]
        except ValueError:
            print('\nMove coordinates in improper format. Try again.\n')
            continue
        if board[move[0], move[1]] != 0:
            print('\nThis field is taken.\n')
            continue
        board[move[0], move[1]] = human_player

        print(f'\nState of the board:\n{board}\n')

        if get_winner(board) == human_player:
            print('You won!')
            break

        if check_if_board_is_full(board, size):
            print('Draw!')
            break

        break

while True:
    computer_move = negamax_alpa_beta_pruned(board, computer_player, -np.inf, np.inf)['move']
    row = computer_move[0]
    col = computer_move[1]
    board[row, col] = computer_player
    print(f'Computer plays: [{row}, {col}]')

    print(f'\nState of the board:\n{board}\n')

    if get_winner(board) == computer_player:
        print('You lost!')
        break

    if check_if_board_is_full(board, size):
        print('Draw!')
        break

    move = input('Play a move. (type e.g. "0, 0" for a move in the left upper corner)\n')

    try:
        move = [int(m) for m in move.split(',')]
    except ValueError:
        print('\nMove coordinates in improper format. Try again.\n')
        continue
    if board[move[0], move[1]] != 0:
        print('\nThis field is taken.\n')
        continue
    board[move[0], move[1]] = human_player

    print(f'\nState of the board:\n{board}\n')

    if get_winner(board) == human_player:
        print('You won!')
        break

    if check_if_board_is_full(board, size):
        print('Draw!')
        break
