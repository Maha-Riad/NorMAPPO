import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tqdm import tqdm
from PPO_learner.ppo import *
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
    wandb.init(entity='mahariad', project='DEPPO', config={
        'env_id': 'cooperativeEnv_separateRewards',
        'env_steps': 9e6,
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
    dataset2 = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    dataset3 = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    cumulative_dataset= TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        
        action, log_probs = ppo.act(states_tensor) #edited by Maha
        action2=[]
        action3=[]

        
        worker_actions=[]
        for a in action:
            worker_action =[]
            worker_action.append(a)
            rand_action2=random.randint(0,8)
            worker_action.append(rand_action2)
            action2.append(rand_action2)
            rand_action3=random.randint(0,8)
            worker_action.append(rand_action3)
            action3.append(rand_action3)
            worker_actions.append(worker_action)
            actions=np.array(worker_actions, dtype=np.int64)
     
        next_states, rewards, done, info = vec_env.step(actions)
        
        rewards1=[]
        rewards2=[]
        rewards3=[]
        cumulative_rewards=[]
        for r in rewards:
            rewards1.append(np.array(r[0:4]))
            rewards2.append(np.array(r[4:8]))
            rewards3.append(np.array(r[8:12]))
            obj1_ret=r[0]+r[4]+r[8]
            obj2_ret=r[1]+r[5]+r[9]
            obj3_ret=r[2]+r[6]+r[10]
            obj4_ret=r[3]+r[7]+r[11]
            cumulative_reward=[obj1_ret,obj2_ret,obj3_ret,obj4_ret]
            
            cumulative_rewards.append(np.array(cumulative_reward))
        
        scalarized_rewards1 = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards1]
        scalarized_rewards2 = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards2]
        scalarized_rewards3 = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards3]
        scalarized_rewards=[]
       
        for env_num in range (len(scalarized_rewards1)):
            scalarized_reward = scalarized_rewards1[env_num]+scalarized_rewards2[env_num]+scalarized_rewards3[env_num]
            scalarized_rewards.append(scalarized_reward)
            
        train_ready = dataset.write_tuple(states, action, scalarized_rewards1, done, log_probs, rewards1)
        dataset2.write_tuple(states, action2, scalarized_rewards2, done, [0,0,0,0,0,0,0,0,0,0,0,0], rewards2)
        dataset3.write_tuple(states, action3, scalarized_rewards3, done, [0,0,0,0,0,0,0,0,0,0,0,0], rewards3)
        cumulative_dataset.write_tuple(states, actions, scalarized_rewards, done, [0,0,0,0,0,0,0,0,0,0,0,0], cumulative_rewards )
       

        print ('The train_ready value is: ',train_ready) #added by Maha for tracing
        if train_ready:
            print('The training is ready') #added by Maha for tracing
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs = dataset.log_objectives()
            for i in range(objective_logs.shape[1]):
                wandb.log({'agent1_Obj_' + str(i): objective_logs[:, i].mean()})
                print ('The objectives mean of agent 1 is: ', objective_logs[:, i].mean())#added by Maha for tracing
            for ret in dataset.log_returns():
                wandb.log({'agent1_Returns': ret})
                print ('The return of agent 1 is' , ret)#added by Maha for tracing
            dataset.reset_trajectories()
            objective_logs2 = dataset2.log_objectives()
            for i in range(objective_logs2.shape[1]):
                wandb.log({'agent2_Obj_' + str(i): objective_logs2[:, i].mean()})
                print ('The objectives mean of agent 2 is:s ', objective_logs2[:, i].mean())#added by Maha for tracing
            for ret in dataset2.log_returns():
                wandb.log({'agent2_Returns': ret})
                print ('The return of agent 2 is' , ret)#added by Maha for tracing
            dataset2.reset_trajectories()
            objective_logs3 = dataset3.log_objectives()
            for i in range(objective_logs3.shape[1]):
                wandb.log({'agent3_Obj_' + str(i): objective_logs3[:, i].mean()})
                print ('The objectives mean of agent 3 is: ', objective_logs3[:, i].mean())#added by Maha for tracing
            for ret in dataset3.log_returns():
                wandb.log({'agent3_Returns': ret})
                print ('The return of agent 3 is' , ret)#added by Maha for tracing
            dataset3.reset_trajectories()

            objective_logs_cum = cumulative_dataset.log_objectives()
            for i in range(objective_logs_cum.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs_cum[:, i].mean()})
                print ('The  cumulative objectives mean is: ', objective_logs_cum[:, i].mean())#added by Maha for tracing
            for ret in cumulative_dataset.log_returns():
                wandb.log({'Returns': ret})
                print ('The cumulative return of is' , ret)#added by Maha for tracing
            cumulative_dataset.reset_trajectories()

        #Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    #vec_env.close()
    torch.save(ppo.state_dict(), 'saved_models/3Agents_L_Z_Random_plotted' + str(config.lambd) + '.pt')
