import random
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import tqdm

import MCTS
from tqdm import trange
from TicTacToe import TicTacToe

from SFQ_sequence import SFQ

#from Model_TicTacToe import ResNet
from Model import ResNet


def selfPlay(game, params, queue, proc_num):
    for _ in range(params['num_self_plays']):
        memory = []
        player = 1
        state = game.get_initial_state()
        model = queue.get(timeout=2)
        # print('PASSED MODEL CPU:',model.state_dict()['startBlock.0.weight'][0][0])
        model.to(torch.device('cuda'))
        # print('PASSED MODEL CUDA:', model.state_dict()['startBlock.0.weight'][0][0])
        mcts = MCTS.MCTS_Play(game=game, model=model, args=params)
        del model
        while True:
            neutral_state = game.change_perspective(state, player)
            action_probs = mcts.search(neutral_state, proc_num)
            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / params['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(game.action_size,
                                      p=temperature_action_probs)  # Divide temperature_action_probs with its sum in case of an error

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                    returnMemory.append((
                        game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                # print(f'\033[94m return memory: {returnMemory} \033[0m')
                queue.put(returnMemory)
                return
                # break

            player = game.get_opponent(player)


def train(memory, neural_net):
    # print(f'\033[36m My memory is {memory} \033[0m')
    random.shuffle(memory)
    for batchIdx in range(0, len(memory), params['batch_size']):
        sample = memory[batchIdx:min(len(memory) - 1, batchIdx + params[
            'batch_size'])]  # Change to memory[batchIdx:batchIdx+self.params['batch_size']] in case of an error
        state, policy_targets, value_targets = zip(*sample)

        state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
            value_targets).reshape(-1, 1)

        state = torch.tensor(state, dtype=torch.float32, device=neural_net.device)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=neural_net.device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32, device=neural_net.device)



        out_policy, out_value = neural_net(state)

        policy_loss = F.cross_entropy(out_policy, policy_targets)
        value_loss = F.mse_loss(out_value, value_targets, reduction='sum')
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'max fidelity: {max(value_targets)}')
    return policy_loss, value_loss, loss


# class Manager(BaseManager):
#     pass

# Manager.register('TicTacToe', TicTacToe)
# Manager.register('MCTS_Play', MCTS.MCTS_Play)
# Manager.register('ResNet',ResNet)

if __name__ == "__main__":
    game = SFQ()
    params = {
        'C': 2,
        'num_searches': 1000,
        'num_iterations': 10,
        'num_self_plays': 200,
        'num_parallel_games': 50,
        'num_epochs': 8,
        'batch_size': 128,
        'temperature': 1.5,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    device = torch.device('cuda')  # 'cuda' if torch.cuda.is_available() else
    print(f'Selected device: {device}')
    neural_net = ResNet(game, 4, 64, device)
    #neural_net.load_state_dict(torch.load('models/SFQ_19.pt'))
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.001, weight_decay=5e-4)

    neural_net.share_memory()

    manager = mp.Manager()
    queue = manager.Queue()
    writer = SummaryWriter()
    for iteration in trange(params['num_iterations']):
        memory = []

        neural_net.eval()
        processes = []

        neural_net.to(torch.device('cpu'))
        for num in range(params['num_parallel_games']):
            queue.put(neural_net)
            proc = mp.Process(target=selfPlay, args=(game, params, queue, num))
            # print(f'\033[92m Proc {num} started \033[0m')
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
            # print(f'\033[96m Proc {proc} joined \033[0m')

        while True:  # seems to be enough to complete your loops, but it's just a demo condition, you should not use this
            try:
                memory += queue.get(timeout=2)
            except Exception as expt:  # the output_queue.get(timeout=1) will wait up to 1 second if the queue is momentarily empty. If the queue is empty for more than 1 sec, it raises an exception and it means the loop is complete. Again, this is not a good condition in real life, and this is just for testing.
                break

        # memory += selfPlay(mcts,game,params,queue)
        neural_net.to(torch.device('cuda'))
        neural_net.train()
        # print('State dict after mp:', neural_net.state_dict()['startBlock.0.weight'][0][0])
        for epoch in trange(params['num_epochs']):
            policy_loss, value_loss, loss = train(memory, neural_net=neural_net)

        print(f'Loss: {loss:.2f}, policy loss: {policy_loss:.2f}, value loss: {value_loss:.2f}')
        writer.add_scalar('Total loss', loss, iteration)
        writer.add_scalar('Policy loss', policy_loss, iteration)
        writer.add_scalar('Value loss', value_loss, iteration)

        torch.save(neural_net.state_dict(), f"models/garbage_new_{iteration}.pt")
        #torch.save(optimizer.state_dict(), f"models/TTTopt_{iteration}.pt")
