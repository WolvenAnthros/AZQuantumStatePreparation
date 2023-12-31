import numpy.random

from TicTacToe import TicTacToe
import numpy as np
import torch

from SFQ_sequence import SFQ
# from Model_TicTacToe import ResNet
from Model import ResNet
import tqdm


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.prior = prior  # probability when a child is initiated, needed for expansion
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        # self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0  # np.sum(self.expandable_moves) == 0 and

    def select(self):
        # UCB score picking
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2  # +1 / 2 bc values can only be +1 and -1 child.value_sum / child.visit_count
        # 1 - because players are changed every turn
        return q_value + self.args['C'] * np.sqrt(self.visit_count / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)  # 1 for player!
                ''''
                we dont change the player, we flip the state so it looks like the opposite player's board
                '''
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prior=prob)
                self.children.append(child)
        #return child
        try:
            return child
        except UnboundLocalError:
            print(f'Warning! No child is created!')
            return None

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    def search(self, states, spGames):
        with torch.no_grad():
            # adding noise to policy (Dirichlet noise)
            # force 1-step exploration
            policy, _ = self.model(
                torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
            )
            policy = torch.softmax(policy, dim=1).cpu().detach().numpy()
            policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                     * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

            # allocate polic to all Self-Play games
            for i, spg in enumerate(spGames):
                spg_policy = policy[i]
                valid_moves = self.game.get_valid_moves(states[i])
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)
                # define a root node

                spg.root = Node(self.game, self.args, states[i], visit_count=1)
                spg.root.expand(spg_policy)
            for search in range(self.args['num_searches']):
                for spg in spGames:
                    spg.node = None
                    node = spg.root

                    while node.is_fully_expanded():
                        node = node.select()

                    value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                    value = self.game.get_opponent_value(value)

                    if is_terminal:
                        node.backpropagate(value)
                    else:
                        spg.node = node

                expandable_spGames = [mapping_idx for mapping_idx in range(len(spGames)) if
                                      spGames[mapping_idx].node is not None]

                if len(expandable_spGames) > 0:
                    states = np.stack([spGames[mapping_idx].node.state for mapping_idx in expandable_spGames])

                    policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                    )
                    # distribution of likelihood
                    policy = torch.softmax(policy, dim=1).cpu().detach().numpy()
                    value = value.cpu().detach().numpy()

                for i, mapping_idx in enumerate(expandable_spGames):
                    node = spGames[mapping_idx].node
                    spg_policy, spg_value = policy[i], value[i]

                    valid_moves = self.game.get_valid_moves(node.state)
                    spg_policy *= valid_moves  # mask invalid moves
                    spg_policy /= np.sum(spg_policy)

                    node = node.expand(spg_policy)
                    node.backpropagate(spg_value)


class MCTS_Play:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    def search(self, state, proc_num):
        with torch.no_grad():
            root = Node(self.game, self.args, state, visit_count=1)

            policy, _ = self.model(
                torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
            )
            policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                     * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
            valid_moves = self.game.get_valid_moves(state)
            policy *= valid_moves
            policy /= np.sum(policy)
            root.expand(policy)

            for search in range(self.args['num_searches']):
                node = root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if not is_terminal:
                    policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                    )

                    policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                    valid_moves = self.game.get_valid_moves(node.state)
                    policy *= valid_moves
                    policy /= np.sum(policy)

                    value = value.item()

                    node.expand(policy)

                node.backpropagate(value)

            action_probs = np.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            return action_probs


if __name__ == '__main__':
    tictactoe = SFQ()
    player = 1

    args = {
        'C': 2,
        'num_searches': 1000,
        'num_iterations': 20,
        'num_self_plays': 500,
        'num_parallel_games': 5,  # number of cores taken by the computation!
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(tictactoe, 4, 64, device=torch.device('cuda'))
    model.load_state_dict(torch.load('models/garbage_new_8.pt'))
    model.eval()
    model = model.to(device)
    mcts = MCTS_Play(tictactoe, args, model)
    state = tictactoe.get_initial_state()

    while True:
        print(state)

        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state)
            #print(f'valid moves: {[i for i in range(tictactoe.action_size) if valid_moves[i] == 1]}')
            #action = int(input(f'{player}: '))
            action = np.random.choice(valid_moves)
            print(f'action {action}')
            # check if the action is valid
            if valid_moves[action] == 0:
                print('action not valid')
                continue
        else:
            neutral_state = tictactoe.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state,None)
            action = np.argmax(mcts_probs)

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

    # pulse_str = '1110-1-1-1-11110-1-1-1-11100-1-1-1111110-1-111111-1-1-1-11111-1-1-101111-1-1-1-1-11111-1-1-1-10111-1-10-10111-1-1-1-1-111111-1-1-111-110-1-1-1-1011-1-1-10111110-1011111-1-1-1-111'
    # pulse_str = pulse_str.replace('1', '1,')
    # pulse_str = pulse_str.replace('0', '0,')
    # pulse_list = pulse_str.split(',')
    # pulse_list.pop(-1)
    # pulse_list = [int(pulse) for pulse in pulse_list]
    # state = np.array(pulse_list)

    # print(state)
    # with tqdm.tqdm(total=125) as bar:
    #     while True:
    #
    #         state_encoded = tictactoe.get_encoded_state(state)
    #         state_encoded = torch.tensor(state_encoded, dtype=torch.float32, device=model.device).unsqueeze(0)
    #         policy, predicted_value = model(state_encoded)
    #
    #         policy = torch.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()
    #         valid_moves = tictactoe.get_valid_moves(state)
    #         policy *= valid_moves
    #         policy /= np.sum(policy)
    #         action = np.random.choice(tictactoe.action_size, p=policy)
    #
    #         # mcts_probs = mcts.search(state,None)
    #         # action = np.argmax(mcts_probs)
    #
    #         bar.update(1)
    #         state = tictactoe.get_next_state(state, action, player)
    #
    #         value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    #
    #         if is_terminal:
    #             print(state)
    #             print(f'Fidelity: {value}')
    #             break
    #
    #         player = tictactoe.get_opponent(player)
    # print(f'pred:{predicted_value}')
