import gym
import numpy as np 
from collections import namedtuple, deque
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make('CartPole-v0')
#env = gym.make('FrozenLake-v0')
GAMMA = 0.99
ENTROPY_BETA = 0.01
nA = env.action_space.n
print('env', env.observation_space.shape[0])
nS = env.observation_space.shape[0]
LR = 3e-2
EPS = np.finfo(np.float32).eps.item()

class Net(nn.Module):
	def __init__(self, s_size = nS, h1_size = 32, h2_size = 32, a_size = nA):
		super(Net, self).__init__()
		self.base = nn.Linear(s_size, h1_size)	

		self.policy = nn.Linear(h1_size, a_size)
		#self.policy2 = nn.Linear(h2_size, a_size)

		self.value = nn.Linear(h1_size, 1)
		#self.value2 = nn.Linear(h2_size, 1)

	def forward(self, x):
		x = F.relu(self.base(x))

		p = self.policy(x)

		v = self.value(x)

		# p = F.relu(self.policy1(x))
		# p = self.policy2(p)

		# v = F.relu(self.value1(x))
		# v = self.value2(v)

		return F.softmax(p, dim=-1), v #policy output, value output

	def select_action(self, state):
			#state = torch.tensor(state)
			state = torch.from_numpy(state).float()
			probs, val = self.forward(state)
			m = Categorical(probs)
			action = m.sample()
			return val, action.item(), m.log_prob(action)

net = Net()
optimizer = optim.Adam(net.parameters(), lr = LR, eps = 1e-3)

def a2c(episodes = 1000, max_timesteps = 1000, render = False):
	total_rewards = []
	for i in range(1, episodes+1):
		state = env.reset()
		episode_rewards = []
		episode_values = []
		saved_log_probs = []
		cur_rewards = 0
		
		for t in range(max_timesteps):
			val, action, log_prob = net.select_action(state)
			#entropy = -np.sum(np.mean(dist) * np.log(dist))
			state, reward, done, _ = env.step(action)
			if render == True:
				env.render()
			episode_rewards.append(reward)
			episode_values.append(val)
			saved_log_probs.append(log_prob)
			cur_rewards += reward
			if done:
				total_rewards.append(cur_rewards)
				break

		#update policy after each episode
		R = 0
		returns = []
		for r in episode_rewards[::-1]:
			R = r + GAMMA * R
			returns.insert(0, R)
		episode_values = torch.tensor(episode_values)
		returns = torch.tensor(returns)
		returns = (returns - returns.mean()) / (returns.std() + EPS)
		
		policy_losses = []
		value_losses = []
		for (log_prob, v, R) in zip(saved_log_probs, episode_values, returns):
			advantage = R - v#.item()
			policy_losses.append(-log_prob * advantage)
			value_losses.append(F.mse_loss(v, torch.tensor([R])))

		optimizer.zero_grad() #zero before start of each backprop
		total_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
		total_loss.backward()
		optimizer.step()

		if i % 10 == 0:
			print('Episode {}\tAverage score: {:.2f}'.format(i, np.mean(total_rewards[-100:])))

		if np.mean(total_rewards[-100:])>=195.0:
			print('Environment solved in {:d} episodes.\tAverage score: {:.2f}'.format(i, np.mean(total_rewards[-100:])))
			break

	return total_rewards

rf = a2c()

fig = plt.figure()
plt.plot(np.arange(1, len(rf)+1), rf)
plt.ylabel('Reward')
plt.xlabel('Episode #')
plt.show()

state = env.reset()
test_eps = 50
for i in range(test_eps):
	state = env.reset()
	for t in range(300):
		_, action, _ = net.select_action(state)
		env.render()
		state, reward, done, _ = env.step(action)
		if done:
			print(t)
			break 

env.close()