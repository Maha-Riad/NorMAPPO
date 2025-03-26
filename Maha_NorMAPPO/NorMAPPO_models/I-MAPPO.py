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

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('The used device is: ',device)

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
    ppo1 = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer1 = torch.optim.Adam(ppo1.parameters(), lr=config.lr_ppo)
    dataset1 = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    ppo2 = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer2 = torch.optim.Adam(ppo2.parameters(), lr=config.lr_ppo)
    dataset2 = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    ppo3 = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer3 = torch.optim.Adam(ppo3.parameters(), lr=config.lr_ppo)
    dataset3 = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    cumulative_dataset= TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        
        action1, log_probs1 = ppo1.act(states_tensor) #edited by Maha
        print ("The action of p is: ",action1) #added by Maha for tracing

        action2, log_probs2 = ppo2.act(states_tensor) #edited by Maha
        print ("The action of l is: ",action2) #added by Maha for tracing

        action3, log_probs3 = ppo3.act(states_tensor) #edited by Maha
        print ("The action of z is: ",action3) #added by Maha for tracing
   
        worker_actions=[]
        for env_num in range (len(action1)):
            worker_action =[]
            worker_action.append(action1[env_num])
            worker_action.append(action2[env_num])
            worker_action.append(action3[env_num])
            worker_actions.append(worker_action)
        actions=np.array(worker_actions, dtype=np.int64)
        print ("The actions list is: ",actions) #added by Maha for tracing
     
        next_states, rewards, done, info = vec_env.step(actions)
        print('The rewards are: ', rewards)  #added by Maha for tracing
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

        scalarized_rewards=[] #added by Maha
       
        for env_num in range (len(scalarized_rewards1)): #added by Maha
            scalarized_reward = scalarized_rewards1[env_num]+scalarized_rewards2[env_num]+scalarized_rewards3[env_num]
            scalarized_rewards.append(scalarized_reward)
 

        train_ready1 = dataset1.write_tuple(states, action1, scalarized_rewards1, done, log_probs1, rewards1)
        train_ready2 = dataset2.write_tuple(states, action2, scalarized_rewards2, done, log_probs2, rewards2)
        train_ready3 = dataset3.write_tuple(states, action3, scalarized_rewards3, done, log_probs3, rewards3)
        cumulative_ready = cumulative_dataset.write_tuple(states, actions, scalarized_rewards, done, [0,0,0,0,0,0,0,0,0,0,0,0], cumulative_rewards )
       
        if train_ready1:
            print('The training of agent 1 is ready') #added by Maha for tracing
            update_policy(ppo1, dataset1, optimizer1, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs1 = dataset1.log_objectives()
            for i in range(objective_logs1.shape[1]):
                wandb.log({'agent1_Obj_' + str(i): objective_logs1[:, i].mean()})
                print ('The objectives mean of agent 1 is: ', objective_logs1[:, i].mean())#added by Maha for tracing
            for ret in dataset1.log_returns():
                wandb.log({'agent1_Returns': ret})
                print ('The return of agent 1 is' , ret)#added by Maha for tracing
            dataset1.reset_trajectories()

        
            
        if train_ready2:
            print('The training of agent 2 is ready') #added by Maha for tracing
            update_policy(ppo2, dataset2, optimizer2, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs2 = dataset2.log_objectives()
            for i in range(objective_logs2.shape[1]):
                wandb.log({'agent2_Obj_' + str(i): objective_logs2[:, i].mean()})
                print ('The objectives mean of agent 2  is: ', objective_logs2[:, i].mean())#added by Maha for tracing
            for ret in dataset2.log_returns():
                wandb.log({'agent2_Returns': ret})
                print ('The return of agent 2 is' , ret)#added by Maha for tracing
            dataset2.reset_trajectories()

        if train_ready3:
            print('The training of agent 3  is ready') #added by Maha for tracing
            update_policy(ppo3, dataset3, optimizer3, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs3 = dataset3.log_objectives()
            for i in range(objective_logs3.shape[1]):
                wandb.log({'agent3_Obj_' + str(i): objective_logs3[:, i].mean()})
                print ('The objectives mean of agent 3  is: ', objective_logs3[:, i].mean())#added by Maha for tracing
            for ret in dataset3.log_returns():
                wandb.log({'agent3_Returns': ret})
                print ('The return of agent 3 is' , ret)#added by Maha for tracing
            dataset3.reset_trajectories()
            
        if cumulative_ready:
            objective_logs_cum = cumulative_dataset.log_objectives()
            for i in range(objective_logs_cum.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs_cum[:, i].mean()})
                print ('The  cumulative objectives mean is: ', objective_logs_cum[:, i].mean())#added by Maha for tracing
            for ret in cumulative_dataset.log_returns():
                wandb.log({'Returns': ret})
                print ('The cumulative return is' , ret)#added by Maha for tracing
            cumulative_dataset.reset_trajectories()
            

        #Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    #vec_env.close()
    torch.save(ppo1.state_dict(), 'saved_models/3PPO_3Datasets_3Rewards_ag1' + str(config.lambd) + '.pt')
    torch.save(ppo2.state_dict(), 'saved_models/3PPO_3Datasets_3Rewards_ag2' + str(config.lambd) + '.pt')
    torch.save(ppo3.state_dict(), 'saved_models/3PPO_3Datasets_3Rewards_ag3' + str(config.lambd) + '.pt')
