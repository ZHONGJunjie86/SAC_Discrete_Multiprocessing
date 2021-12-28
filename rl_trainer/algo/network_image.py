from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
import torch.nn as nn
import torch.nn.functional as F
import torch

HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()   #20*10
        self.conv1 = nn.Conv2d(4,16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 111032 -> 8*10*32

        self.linear_1 = nn.Linear(1280, 12) #14464 = 3584 1792
        #self.linear_2 = nn.Linear(128,12)
        

    def forward(self, tensor_cv): #,batch_size
        # CV
        x = F.relu(self.conv1(tensor_cv))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=  x.reshape(x.size()[0],1,1280) #1792 7 68
        batch_size = x.size()[0]
        #print("x.size()",x.size())
        #x= x.repeat(1,3,1)
        #print("after x.size()",x.size())
        #x = F.relu(self.linear_CNN_1(x)) #.reshape(x.size()[0],3,128)
        #action1 = F.relu(self.linear_CNN_2(x))
        #action = F.relu(self.linear_CNN_3(x))+1e-5
        #action = self.linear_CNN_3(x)   #.clamp(1e-10,1e10)

        #x = F.relu(self.linear_1(x)) #.reshape(x.size()[0],3,128)
        action = torch.tanh(self.linear_1(x)).reshape(batch_size,3,4)

        return action


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()
        self.conv1_1 = nn.Conv2d(4,16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2_1 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4_1 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 16632 -> 14464
        self.linear_CNN_1_1 = nn.Linear(1280, 128) #14464 = 3584  1792
        #
        self.linear_1_1 = nn.Linear(12, 32)#32*3=96
        self.linear_2_1 = nn.Linear(32, 128)
        #
        self.linear_4_1 = nn.Linear(256,3) #64

        ##########################################
        self.conv1_2 = nn.Conv2d(4,16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4_2 = nn.Conv2d(16,16, kernel_size=(6,3), stride=1, padding=1) # 16632 -> 14464
        self.linear_CNN_1_2 = nn.Linear(1280, 128) #14464 = 3584 1792
        #
        self.linear_1_2 = nn.Linear(12, 32)#32*3=96
        self.linear_2_2 = nn.Linear(32, 128)
        #
        self.linear_4_2 = nn.Linear(256,3) #512


    def forward(self, tensor_cv, action_batch):
        # CV
        x = F.relu(self.conv1_1(tensor_cv))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv4_1(x))
        #
        batch_size = x.size()[0]
        x=  x.reshape(batch_size,1,1280) #1792
        x = F.relu(self.linear_CNN_1_1(x))#.reshape(x.size()[0],3,128)
        #
        action_batch = action_batch.reshape(batch_size,1,12)
        y = F.relu(self.linear_1_1(action_batch))
        y = F.relu(self.linear_2_1(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        out_1 = torch.tanh(self.linear_4_1(z)).reshape(batch_size,3,1)
        #######################################################
        # CV/home/j-zhong/work_place/TD3_SAC_PPO_multi_Python/rl_trainer/TD3_history.png
        x = F.relu(self.conv1_2(tensor_cv))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv4_2(x))
        #
        x=  x.reshape(batch_size,1,1280)#.repeat(1,3,1) 1792
        x = F.relu(self.linear_CNN_1_2(x))#.reshape(x.size()[0],3,128)
        #
        action_batch = action_batch.reshape(batch_size,1,12)
        y = F.relu(self.linear_1_2(action_batch))
        y = F.relu(self.linear_2_2(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        out_2 = torch.tanh(self.linear_4_2(z)).reshape(batch_size,3,1)

        return out_1,out_2





