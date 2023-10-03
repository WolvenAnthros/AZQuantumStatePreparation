import random
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

import MCTS
from tqdm import trange

from SFQ_sequence import SFQ
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
                                      p=temperature_action_probs)

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
                queue.put(returnMemory)
                return

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
    return policy_loss, value_loss, loss, max(value_targets).squeeze(0).cpu().detach().numpy()


if __name__ == "__main__":
    game = SFQ()
    params = {
        'C': 4,
        'num_searches': 300,
        'num_iterations': 20,
        'num_self_plays': 300,
        'num_parallel_games': 2, # number of cores
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 2,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    device = torch.device('cuda')  # 'cuda' if torch.cuda.is_available() else
    print(f'Selected device: {device}')
    neural_net = ResNet(game, 4, 64, device)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.95,verbose=True)

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

        while True:
            try:
                memory += queue.get(timeout=2)
            except Exception as expt:  # the output_queue.get(timeout=1) will wait up to 1 second if the queue is momentarily empty. If the queue is empty for more than 1 sec, it raises an exception and it means the loop is complete.
                break

        neural_net.to(torch.device('cuda'))
        neural_net.train()
        # print('State dict after mp:', neural_net.state_dict()['startBlock.0.weight'][0][0])
        for epoch in trange(params['num_epochs']):
            policy_loss, value_loss, loss, max_fidelity = train(memory, neural_net=neural_net)
        scheduler.step()

        print(f'Loss: {loss:.2f}, \n policy loss: {policy_loss:.2f},  \n value loss: {value_loss:.2f}, \n \033[36m max_fidelity:{max_fidelity:.3f} \033[0m')
        writer.add_scalar('Total loss', loss, iteration)
        writer.add_scalar('Policy loss', policy_loss, iteration)
        writer.add_scalar('Value loss', value_loss, iteration)
        writer.add_scalar('Max encountered fidelity', max_fidelity, iteration)

        torch.save(neural_net.state_dict(), f"models/garbage_new_{iteration}.pt")
        #torch.save(optimizer.state_dict(), f"models/TTTopt_{iteration}.pt")
