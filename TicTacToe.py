import numpy as np


class TicTacToe:
    '''
    "move" function is separated with get_next_state, check_win, get_value_and_terminated
    '''

    def __init__(self):
        self.row_count = 5
        self.column_count = 5
        self.action_size = self.row_count * self.column_count  # actions correspond to board grid

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        # int between 0 - 8 (1-9 cell)
        # encode actions
        row = action // self.column_count
        column = action % self.column_count

        # a player's turn is described as his deployment on a certain cell

        state[row, column] = player  # 1/-1
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)  # element-wise comparison

    def check_win(self, state, action):
        if action == None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
                np.sum(state[row, :]) == player * self.column_count  # 3 in a row
                or np.sum(state[:, column]) == player * self.row_count  # 3 in a column
                or np.sum(np.diag(state)) == player * self.row_count  # diagonal
                or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count  # opposite diagonal
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:  # from matrix to tensor
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state


if __name__ == '__main__':
    tictactoe = TicTacToe()
    player = 1

    state = tictactoe.get_initial_state()

    while True:
        print(state)
        valid_moves = tictactoe.get_valid_moves(state)
        print(f'valid moves: {[i for i in range(tictactoe.action_size) if valid_moves[i] == 1]}')
        action = int(input(f'{player}: '))

        # check if the action is valid
        if valid_moves[action] == 0:
            print('action not valid')
            continue

        state = tictactoe.get_next_state(state, action, player)

        value, is_terminal = tictactoe.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = tictactoe.get_opponent(player)
