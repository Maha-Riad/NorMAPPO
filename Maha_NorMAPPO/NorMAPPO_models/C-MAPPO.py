#The aim of this file is to use 1 PPO network to train 3 agents at a time
#i.e. using 1 network and 1 cumulative reward

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tqdm import tqdm
from PPO_learner.jointAction_ppo import *
import torch
from envs.gym_wrapper import *
import wandb
import argparse
import time


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Fetch ratio args
    #parser = argparse.ArgumentParser(description='Preference Lambda.')
    #parser.add_argument('--lambd', nargs='+', type=float)
    #args = parser.parse_args()

    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': 'cooperativeEnv_GlobalReward',
        'env_steps': 90,#e6, # this variable can increase to 18e6 to see its learning 
        'batchsize_ppo': 12,
        'n_workers': 12,
        'lr_ppo': 3e-4, #determines the gradient step size used in the agent's Adam optimizer, a trust region clip paramete
        'entropy_reg': 0.05,
        'lambd': [2, 1, 1, 1],
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        
        actions, log_probs = ppo.act(states_tensor) 

        next_states, rewards, done, info = vec_env.step(actions)
        scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]
        actions=ppo.revertJointActions(actions)# added by Maha -- this should be only executed in case of joined actions
             
        train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs, rewards)
        if train_ready:
            print('The training is ready') #added by Maha for tracing
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs = dataset.log_objectives()
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
                print ('The objectives mean is: ', objective_logs[:, i].mean())#added by Maha for tracing
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
                print ('The return is' , ret)#added by Maha for tracing
            dataset.reset_trajectories()

        #Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    #vec_env.close()
    torch.save(ppo.state_dict(), 'saved_models/3Ag_jointAction_' + str(config.lambd) + '.pt')
