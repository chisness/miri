import gym
import collections
import numpy as np

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.95
ALPHA = 0.1
# EPSILON_START = 0.7
# EPSILON_DECAY = 0.99
# EPSILON_FINAL = 0.1
# EPSILON = EPSILON_START
EPSILON = 0.1
NUM_EPISODES = 1000
TEST_EPISODES = 100

env = gym.make(ENV_NAME)
print(env.action_space.n)
q_table = np.zeros([env.observation_space.n, env.action_space.n]) #state, action
print(q_table)

for episode in range(NUM_EPISODES):
	state = env.reset()
	done = False 
	while not done:
		dart = np.random.rand()
		if dart < EPSILON: #explore with random action
			action = env.action_space.sample()
		else:
			best_actions = np.argwhere(q_table[state] == np.max(q_table[state]))
			action = np.random.choice(best_actions.flatten())
		# if EPSILON > EPSILON_FINAL:
		# 	EPSILON *= EPSILON_DECAY
		print(action)
		print(q_table)
		next_state, reward, done, info = env.step(action)
		print('reward', reward)
		prev_q_val = q_table[state, action]
		print(prev_q_val)
		max_val_next_state = np.max(q_table[next_state])
		new_q_val = prev_q_val + ALPHA*(reward + GAMMA*(max_val_next_state) - prev_q_val)
		
		q_table[state, action] = new_q_val
		state = next_state

	if episode % 100 == 0:
		print(f'Episode {episode}')
		print(q_table)
		env.render()


for test_episode in range(TEST_EPISODES):
	reward_sum = 0
	for i in range(100):
		state = env.reset()
		done = False
		while not done: 
			action = np.argmax(q_table[state])
			next_state, reward, done, info = env.step(action)
			reward_sum += reward
			state = next_state 
	print(reward_sum / 100*1.)