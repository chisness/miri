import gym
import numpy as np 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make('CartPole-v0')

GAMMA = 0.99
nA = env.action_space.n
nS = env.observation_space.shape[0]
LR = 3e-2
EPS = np.finfo(np.float32).eps.item()
test_eps = 50

class Net(nn.Module):
	def __init__(self, s_size = nS, h_size = 32, a_size = nA):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(s_size, h_size)
		self.fc2 = nn.Linear(h_size, a_size)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

	def select_action(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		probs = self.forward(state)
		m = Categorical(probs)
		action = m.sample()
		return action.item(), m.log_prob(action)

net = Net()
optimizer = optim.Adam(net.parameters(), lr = LR)

def vpg(episodes = 1000, max_timesteps = 1000, render = False):
	total_rewards = []
	for i in range(1, episodes+1):
		state = env.reset()
		episode_rewards = []
		saved_log_probs = []
		cur_rewards = 0

		#play each episode and save rewards and log_probs
		for t in range(max_timesteps):
			action, log_prob = net.select_action(state)
			saved_log_probs.append(log_prob)
			state, reward, done, _ = env.step(action)
			if render == True:
				env.render()
			episode_rewards.append(reward)
			cur_rewards += reward
			if done:
				total_rewards.append(cur_rewards)
				break

		#update policy after each episode
		R = 0 #rewards to go
		returns = []
		# print(episode_rewards)
		for r in episode_rewards[::-1]:
			R = r + GAMMA * R
			returns.insert(0, R)
		returns = torch.tensor(returns)
		returns = (returns - returns.mean()) / (returns.std() + EPS)

		policy_loss = []
		for (log_prob, exp_reward) in zip(saved_log_probs, returns):
			policy_loss.append(-log_prob * exp_reward)
			#negative because we are doing gradient ascent but pytorch doing descent 
		policy_loss = torch.cat(policy_loss).mean()

		optimizer.zero_grad() #zero before start of each backprop
		policy_loss.backward()
		optimizer.step()

		if i % 10 == 0:
			print('Episode {}\tAverage score: {:.2f}'.format(i, np.mean(total_rewards[-100:])))

		if np.mean(total_rewards[-100:])>=195.0:
			print('Environment solved in {:d} episodes.\tAverage score: {:.2f}'.format(i, np.mean(total_rewards[-100:])))
			break

	return total_rewards



rf = vpg()

fig = plt.figure()
plt.plot(np.arange(1, len(rf)+1), rf)
plt.ylabel('Reward')
plt.xlabel('Episode #')
plt.show()

for i in range(test_eps):
	state = env.reset()
	for t in range(300):
		action, _ = net.select_action(state)
		env.render()
		state, reward, done, _ = env.step(action)
		if done:
			print(t)
			break 