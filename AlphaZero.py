import concurrent
import random
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import tqdm
from concurrent.futures import ProcessPoolExecutor as Pool
import concurrent.futures


import MCTS
from tqdm import trange
from TicTacToe import TicTacToe
from SFQ_sequence import SFQ
from Model import ResNet
#from Model_TicTacToe import ResNet

writer = SummaryWriter()


class AlphaZero():
    def __init__(self, model, mcts, optimizer, game, params):
        self.args = params
        self.game = game
        self.optimizer = optimizer
        self.model = model
        self.mcts = mcts  # MCTS.MCTS(game, params, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        # state = self.game.get_initial_state()
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:  # instead of while True

            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)  # FIXME: dont forget this

            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                # add temperature
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])

                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append(
                            (self.game.get_encoded_state(hist_neutral_state),
                             hist_action_probs,
                             hist_outcome)
                        )
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        # shuffle train data

        mean_policy_loss, mean_val_loss, mean_loss = 0, 0, 0
        loss_index = 0
        max_value = 0
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory) - 1,
                                          batch_idx + self.args[
                                              'batch_size'])]  # min len to not exceed memory buffer
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            if value_targets[0][0] > max_value:
                max_value = value_targets[0][0]
                print(f'max fidelity: {max_value}')

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # predict the state
            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            print(f'Loss: {loss:.2f}, policy loss: {policy_loss:.2f}, value loss: {value_loss:.2f}')
            mean_policy_loss += policy_loss
            mean_val_loss += value_loss
            mean_loss += loss
            loss_index += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return mean_policy_loss / loss_index, mean_val_loss / loss_index, mean_loss / loss_index

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            # loop over self-play games
            self.model.eval()

            num_processes = self.args['num_parallel_games']

            # with tqdm.tqdm(total=num_processes) as progress_bar:
            #     with Pool(max_workers=num_processes) as executor:
            #         results = [executor.submit(self.selfPlay) for i in range(num_processes)]
            #         for f in concurrent.futures.as_completed(results):
            #             memory += f.result()
            #             progress_bar.update(1)

            for _ in trange(num_processes):
                memory += self.selfPlay()

            self.model.train()

            for epoch in trange(self.args['num_epochs']):
                policy_loss, value_loss, loss = self.train(memory)

                writer.add_scalar('Policy loss', policy_loss, iteration + epoch)
                writer.add_scalar('Value loss', value_loss, iteration + epoch)
                writer.add_scalar('Total loss', loss, iteration + epoch)

            torch.save(self.model.state_dict(), f"models/modelAZ_{iteration}.pt")
            #torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}.pt")


# parallel information storage
class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None


if __name__ == '__main__':
    tictactoe = SFQ()

    device = torch.device('cuda')  # 'cuda' if torch.cuda.is_available() else
    print(f'Selected device: {device}')
    model = ResNet(tictactoe, 4, 64, device)
    #model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-4)
    args = {
        'C': 2,
        'num_searches': 10,  # 00
        'num_iterations': 8,
        'num_self_plays': 5,  # 00
        'num_parallel_games': 3,  # number of cores taken by the computation!
        'num_epochs': 8,  # 4
        'batch_size': 128,
        'temperature': 1,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }


    mcts = MCTS.MCTS(game=tictactoe, model=model, args=args)
    alphazero = AlphaZero(model=model, mcts=mcts, optimizer=optimizer, game=tictactoe, params=args)

    alphazero.learn()

    # with tqdm.tqdm(total=num_processes) as progress_bar:
    #     with Pool(max_workers=num_processes) as executor:
    #         results = [executor.submit(self.selfPlay) for i in range(num_processes)]
    #         for f in concurrent.futures.as_completed(results):
    #             memory += f.result()
    #             progress_bar.update(1)
