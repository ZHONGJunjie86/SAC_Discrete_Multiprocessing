import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.network_image import Actor, Critic
import torch.nn.functional as F

class TD3:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation

        # Initialise actor network and critic network with ξ and θ
        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr*2)

        # Initialise target network and critic network with ξ' ← ξ and θ' ← θ
        self.actor_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.c_loss = 0
        self.a_loss = 0

        self.total_it = 0
        self.policy_noise = 0.05 #*self.a_lr*10000 #std
        self.noise_clip = 0.1  #*self.a_lr*10000
        self.policy_freq = 2

        self.training_times = 1

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):
        obs = torch.Tensor([obs]).to(self.device)
        action = self.actor(obs)
        action = (action+1).clamp(1e-6,2)
        return action.cpu().detach().numpy()[0]

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))

    def update(self,new_lr):

        if new_lr != self.a_lr:
            self.a_lr = new_lr
            self.c_lr = new_lr*2
            #self.training_times = 100
            #self.policy_noise = 0.5*self.a_lr*10000 #std
            #self.noise_clip = 0.5*self.a_lr*10000
            
            print("new_lr",new_lr)#,"self.noise_clip",self.noise_clip)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),  self.a_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  self.c_lr*3)

        if len(self.replay_buffer) < 25e3:#self.batch_size:
            return 0, 0

        k = 1.0 + len(self.replay_buffer) / 1e6
        #batch_size_ = int(self.batch_size * k)
        self.training_times = int(k * 100)
        
        for i in range(self.training_times): #self.training_times
            #self.total_it += 1
            # Sample a greedy_min mini-batch of M transitions from R
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

            state_batch = torch.Tensor(state_batch).reshape(self.batch_size,4,20,10).to(self.device)
            action_batch = torch.Tensor(action_batch)
            #print("action_batch.size()",action_batch.size())
            action_batch = action_batch.reshape(self.batch_size,3,4).to(self.device) #4,-1
            reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
            next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size,4,20,10).to(self.device)
            done_batch = torch.Tensor(done_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
            #print("action_batch,reward_batch,done_batch\n",action_batch.size(),reward_batch.size(),done_batch.size())

            # Compute target value for each agents in each transition using the Bi-RNN
            with torch.no_grad():
                noise = (
                    torch.randn_like(action_batch) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                #print("noise.size()",noise.size(),"\nself.actor_target(next_state_batch).size()",self.actor_target(next_state_batch).size())
                target_next_actions = (
                    self.actor_target(next_state_batch) + noise
                ).clamp(-1, 1)

                target_next_q_1,target_next_q_2 = self.critic_target(next_state_batch, target_next_actions)
                target_next_q = torch.min(target_next_q_1,target_next_q_2)
                #print("target_next_q",target_next_q.size())
                q_hat = reward_batch + self.gamma * target_next_q * (1 - done_batch)
                #print("q_hat",q_hat.size())

            # Compute critic gradient estimation according to Eq.(8)
            current_Q1,current_Q2 = self.critic(state_batch, action_batch)
            #print("current_Q1,current_Q2",current_Q1.size(),current_Q2.size())
            #print("current_Q1.size()",current_Q1.size()," q_hat.size()",q_hat.size())
            loss_critic = torch.nn.SmoothL1Loss()(current_Q1, q_hat) +  torch.nn.SmoothL1Loss()(current_Q2, q_hat)
            #F.mse_loss(current_Q1, q_hat) + F.mse_loss(current_Q2, q_hat)
            #torch.nn.MSELoss()(q_hat, main_q)

            # Update the critic networks based on Adam
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            #clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

            # Update the actor networks based on Adam
            # Delayed policy updates
            if i% self.policy_freq == 0: #i self.training_times 
                #print("train Policy")
                #self.total_it = 0
                # Compute actor gradient estimation according to Eq.(7)
                # and replace Q-value with the critic estimation
                x,_ = self.critic(state_batch, self.actor(state_batch))
                loss_actor = -x.mean()
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                #clip_grad_norm_(self.actor.parameters(), 1)
                self.actor_optimizer.step()

                self.a_loss += loss_actor.item()
                self.c_loss += loss_critic.item()

                # Update the target networks
                #soft_update(self.critic, self.critic_target, self.tau)
                #soft_update(self.actor, self.actor_target, self.tau)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)
        #base_path: D:\Software\PythonWork\Competition\TD3_SAC_PPO_multi_Python\rl_trainer\models\snakes_3v3\run1\trained_model

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        print("---------------save-------------------")
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path:",base_path)
        print("new_lr: ",self.a_lr)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_4000"  + ".pth") #+ str(episode)
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_4000" + ".pth") #+ str(episode) 
        torch.save(self.critic.state_dict(), model_critic_path)
