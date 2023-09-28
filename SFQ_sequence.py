import numpy as np
from args import args
from SFQ_calc import reward_calculation

max_sequence_length = args['pulse_array_length']
config = args['qbit_simulation_config']
reward_threshold = args['reward_threshold']
polarities_num = args['number_of_polarities']
unexplored_pulse = 9


class SFQ:
    '''
    "move" function is separated with get_next_state, check_win, get_value_and_terminated
    '''

    def __init__(self):
        self.length = max_sequence_length *2
        self.action_size = max_sequence_length * polarities_num  # actions correspond to board grid

    def get_initial_state(self):
        return np.full(self.length, unexplored_pulse, dtype=int)

    def get_next_state(self, state, action, player):
        # int between 0 - 8 (1-9 cell)
        # encode actions
        index = action // polarities_num
        true_action = action % polarities_num
        state_copy = state.copy()
        state_copy[index + int(
            bool(player - 1)) * max_sequence_length] = true_action - 1  # if action=0 -> 0-1 = -1 -> actual action
        return state_copy

    def get_valid_moves_per_state(self, state):
        allowed_pulses = np.repeat(state, polarities_num)
        allowed_pulses = np.array([1 if x == unexplored_pulse else 0 for x in allowed_pulses])
        actions_list = np.array([1 for x in range(max_sequence_length * polarities_num)])
        return actions_list * allowed_pulses

    def get_valid_moves(self, state):
        valid_moves_1, valid_moves_2 = self.get_valid_moves_per_state(
            state[:max_sequence_length]), self.get_valid_moves_per_state(state[max_sequence_length:])
        return (valid_moves_1 + valid_moves_2 != 0).astype(np.uint8)

    def get_value_and_terminated(self, state, action):
        if np.sum(self.get_valid_moves(state)) == 0:
            reward_1 = reward_calculation(state[:max_sequence_length])
            reward_2 = reward_calculation(state[max_sequence_length:])
            print(f'Fidelity_1:{reward_1:.3f}, Fidelity_2:{reward_2:.3f}')
            if reward_1 > reward_2:
                return 1, True
            elif reward_2 > reward_1:
                return -1, True
            else:
                return 0, True
        else:
            return 0, False

    # def get_value_and_terminated(self, state, action):
    #     if np.sum(self.get_valid_moves(state)) == 0:
    #         return reward_calculation(state), True
    #     return 0, False  # FIXME: maybe we should return ordinary reward value?

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return value

    def change_perspective(self, state, player):
        # if player ==1:
        #     return state[:max_sequence_length]
        # else:
        #     return state[max_sequence_length:]
        return state

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1, state == unexplored_pulse)
        ).astype(np.float32)

        if len(state.shape) == 2:  # from matrix to tensor
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state


if __name__ == '__main__':
    tictactoe = SFQ()
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
            print(
                f'Fidelity_1 {reward_calculation(state[:max_sequence_length])} \nFidelity_2 {reward_calculation(state[max_sequence_length:])}\nReward:{value}')
            break

        player = tictactoe.get_opponent(player)
