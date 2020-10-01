#actions hit or stick
#state is player's sum and dealer's showing card and if card is 0
#cards from an infinite deck
#rewards +1, 0, -1
import gym
import numpy as np 
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections 

from gym.envs.registration import register

#how to get the natural and double_down set to true but default false

register(id='BlackjackMax-v0', entry_point='blackjack1:BlackjackEnv1')

ENV_NAME = "BlackjackMax-v0"
EPSILON = 0.1

DOUBLE = 2
HIT = 1
STICK = 0
dd = False
if dd:
	BETS = [STICK, HIT, DOUBLE]
else:
	BETS = [STICK, HIT]

class Player:
	def __init__(self):
		self.env = gym.make(ENV_NAME, natural=False, double_down = dd, dealer_expose = True)
		self.state = self.env.reset()
		self.policy = collections.defaultdict(self.default_policy)
		self.values = collections.defaultdict(float)

	def default_policy(self):
		return [1/len(BETS)] * len(BETS)

	def play_action(self, blackjack_state):
		player_sum, dealer_upcard, usable_ace = blackjack_state
		#return STICK if player_sum >= 20 else HIT
		return self.policy[(blackjack_state)]

	def play_action_argmax(self, blackjack_state):
		return np.argmax(self.policy[(blackjack_state)])


if __name__ == "__main__":
	agent = Player()
	new_state = agent.state
	print(new_state)
	#returns = collections.defaultdict(list)
	returns_sum = collections.defaultdict(float)
	returns_count = collections.defaultdict(float)
	episode = []
	agent.env.reset()
	#print('default', agent.default_policy())

	for i in range(10000000):
		while True:
			# print('state', new_state)
			action_probs = agent.play_action(new_state)
			#print('action probs', action_probs)
			action = np.random.choice(BETS, p=action_probs)
			#print('action', action)
			#action = agent.play_action(new_state)
			#print('action', action)
			episode.append((new_state, action))
			new_state, reward, done, _ = agent.env.step(action)
			#print('reward', reward)
			#print('done', done)
			#print('episode', episode)

			if done:
				# print('DONE!!', reward)
				# print('episode', episode)
				for (state, action) in episode:
					returns_sum[(state,action)] += reward
					returns_count[(state,action)] += 1
					agent.values[(state, action)] = returns_sum[(state,action)] / returns_count[(state,action)]
				
				for (state, _) in episode:
					vals = [agent.values[(state, a)] for a in BETS]
					best_action = np.argmax(vals)
					# best_action = None
					# best_action_value = -100
					# for a in BETS:
					# 	#print('action', a)
					# 	#print('value state action', agent.values[(state, a)])
					# 	#print('best action value', best_action_value)
					# 	if agent.values[(state, a)] > best_action_value:
					# 		best_action_value = agent.values[(state, a)]
					# 		best_action = a
					# print('state', state)
					for a in BETS:
						if a == best_action:
							#print('best action', best_action)
							#print('action', a)
							#agent.policy[state][a] = 1
							agent.policy[state][a] = 1 - EPSILON + EPSILON/len(BETS)
						else:
							#agent.policy[state][a] = 0
							agent.policy[state][a] = EPSILON/len(BETS)
						

				#print('policy', agent.policy)
				episode = []
				new_state = agent.env.reset()
				if i % 100000 == 0:
					print(i)
				# print('START NEW!!')
				break

	# EVALUATE POLICY
	eval_pol_reward = 0
	eval_it = 5000000
	for i in range(eval_it):
		print('iteration', i)
		new_state = agent.env.reset()
		while True:
			action = agent.play_action_argmax(new_state)
			new_state, reward, done, _ = agent.env.step(action)
			# print(reward)
			if done:
				# print('total reward', eval_pol_reward)
				eval_pol_reward += reward
				# print('total reward after', eval_pol_reward)
				new_state = agent.env.reset()
				break

	avg_win = eval_pol_reward/eval_it
	print('avg_win', avg_win)



	#print(sorted(agent.values.items(), key=lambda k: agent.values[k[0]]))
	#print(agent.policy)
	# print(agent.values)

	# s = collections.defaultdict(str)
	# pi = np.zeros((17, 17))
	# for dealer in range(5,22):
	# 	print('dealer', dealer)
	# 	for player in range(5,22):
	# 		best_action = np.argmax(agent.policy[(player,dealer,False)])
	# 		#print('policy', agent.policy[(player,dealer,False)])
	# 		pi[player-5, dealer-5] = best_action #our card, dealer card, valid Ace
	# 		if best_action == 0:
	# 			s[player-5, dealer-5] = 'S'
	# 		elif best_action == 1:
	# 			s[player-5, dealer-5] = 'H'
	# 		else:
	# 			s[player-5, dealer-5] = 'D'
	# 		print('Dealer showing: {}, Player total: {}, Best action: {})\n'.format(dealer, player, best_action))

	# print(pi)
	# fig, ax = plt.subplots()
	# im = ax.imshow(pi, extent=[4.5, 21.5, 21.5, 4.5])
	# ax.set_xticks(np.arange(5,22))
	# ax.set_yticks(np.arange(5,22))
	# ax.set_xlabel('Dealer showing')
	# ax.set_ylabel('Player hand')
	# for i in range(5,22):
	# 	for j in range(5,22):
	# 		text = ax.text(i, j, s[j-5, i-5], ha="center", va="center")
	# ax.set_xticklabels(np.arange(5,22))
	# ax.set_yticklabels(np.arange(5,22))

	# plt.show()

	# figure = plt.figure()
	# ax = plt.axes(projection = '3d')
	# x = np.arange(5,22)
	# y = np.arange(12, 22)
	# plt.xlabel('Dealer showing')
	# plt.ylabel('Player sum')
	# plt.title('Value with no usable ace')
	# z = np.zeros((10,17))
	# for i in range(5,22):
	# 	for j in range(12,22):
	# 		z[j-12, i-5] = agent.values[(j, i, False), np.argmax(agent.policy[(j,i,False)])] #our card, dealer card, valid Ace
	# x,y = np.meshgrid(x,y)
	# ax.set_xticks(np.arange(5,22))
	# ax.set_yticks(np.arange(12,22))
	# ax.set_xticklabels(np.arange(5,22))
	# ax.set_yticklabels(np.arange(12,22))
	# ax.set_zlim(-1,1)
	# ax.plot_surface(x,y,z)
	# plt.show()

	# figure = plt.figure()
	# ax = plt.axes(projection = '3d')
	# x = np.arange(5,22)
	# y = np.arange(12, 22)
	# plt.xlabel('Dealer showing')
	# plt.ylabel('Player sum')
	# plt.title('Value with usable ace')
	# z = np.zeros((10,17))
	# for i in range(5,22):
	# 	for j in range(12,22):
	# 		z[j-12, i-5] = agent.values[(j, i, True), np.argmax(agent.policy[(j,i,True)])] #our card, dealer card, valid Ace
	# x,y = np.meshgrid(x,y)
	# ax.set_xticks(np.arange(5,22))
	# ax.set_yticks(np.arange(12,22))
	# ax.set_xticklabels(np.arange(5,22))
	# ax.set_yticklabels(np.arange(12,22))
	# ax.set_zlim(-1,1)
	# ax.plot_surface(x,y,z)
	# plt.show()

	# s = collections.defaultdict(str)
	# pi = np.zeros((14, 10))
	# for dealer in range(1,11):
	# 	# print('dealer', dealer)
	# 	for player in range(8,22):
	# 		best_action = np.argmax(agent.policy[(player,dealer,False)])
	# 		#print('policy', agent.policy[(player,dealer,False)])
	# 		pi[player-8, dealer-1] = best_action #our card, dealer card, valid Ace
	# 		if best_action == 0:
	# 			s[player-8, dealer-1] = 'S'
	# 		elif best_action == 1:
	# 			s[player-8, dealer-1] = 'H'
	# 		else:
	# 			s[player-8, dealer-1] = 'D'
	# 		#print('Dealer showing: {}, Player total: {}, Best action: {})\n'.format(dealer, player, best_action))

	# print(pi)
	# pi = np.flipud(pi)
	# fig, ax = plt.subplots()
	# im = ax.imshow(pi, extent=[0.5, 10.5, 7.5, 21.5])
	# ax.set_xticks(np.arange(1,11))
	# ax.set_yticks(np.arange(8,22))
	# ax.set_title('Optimal strategy no usable Ace')
	# ax.set_xlabel('Dealer showing')
	# ax.set_ylabel('Player sum')
	# for i in range(1,11):
	# 	for j in range(8,22):
	# 		text = ax.text(i, j, s[j-8, i-1], ha="center", va="center")
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# ax.set_yticklabels(np.arange(8,22))

	# plt.show()

	# s = collections.defaultdict(str)
	# pi = np.zeros((10, 10))
	# for dealer in range(1,11):
	# 	# print('dealer', dealer)
	# 	for player in range(12,22):
	# 		best_action = np.argmax(agent.policy[(player,dealer,True)])
	# 		pi[player-12, dealer-1] = best_action #our card, dealer card, valid Ace
	# 		if best_action == 0:
	# 			s[player-12, dealer-1] = 'S'
	# 		elif best_action == 1:
	# 			s[player-12, dealer-1] = 'H'
	# 		else:
	# 			s[player-12, dealer-1] = 'D'
	# 		#print('Dealer showing: {}, Player total: {}, Best action: {})\n'.format(dealer, player, best_action))

	# print(pi)
	# pi = np.flipud(pi)
	# fig, ax = plt.subplots()
	# im = ax.imshow(pi, extent=[0.5, 10.5, 11.5, 21.5])
	# ax.set_xticks(np.arange(1,11))
	# ax.set_yticks(np.arange(12,22))
	# ax.set_title('Optimal strategy usable Ace')
	# ax.set_xlabel('Dealer showing')
	# ax.set_ylabel('Player sum')
	# for i in range(1,11):
	# 	for j in range(12,22):
	# 		text = ax.text(i, j, s[j-12, i-1], ha="center", va="center")
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# ax.set_yticklabels(np.arange(12,22))

	# plt.show()


	# figure = plt.figure()
	# ax = plt.axes(projection = '3d')
	# x = np.arange(1,11)
	# y = np.arange(8, 22)
	# plt.xlabel('Dealer showing')
	# plt.ylabel('Player sum')
	# plt.title('Optimal value with no usable Ace')
	# z = np.zeros((14, 10))
	# for i in range(1,11):
	# 	for j in range(8,22):
	# 		z[j-8, i-1] = agent.values[(j, i, False), np.argmax(agent.policy[(j,i,False)])] #our card, dealer card, valid Ace
	# x,y = np.meshgrid(x,y)
	# ax.set_xticks(np.arange(1,11))
	# ax.set_yticks(np.arange(8,22))
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# ax.set_yticklabels(np.arange(8,22))
	# ax.set_zlim(-1,1)
	# ax.plot_surface(x,y,z)
	# plt.show()


	# figure = plt.figure()
	# ax = plt.axes(projection = '3d')
	# x = np.arange(1,11)
	# y = np.arange(12, 22)
	# plt.xlabel('Dealer showing')
	# plt.ylabel('Player sum')
	# plt.title('Optimal value with usable Ace')
	# z = np.zeros((10, 10))
	# for i in range(1,11):
	# 	for j in range(12,22):
	# 		z[j-12, i-1] = agent.values[(j, i, True), np.argmax(agent.policy[(j,i,True)])] #our card, dealer card, valid Ace
	# x,y = np.meshgrid(x,y)
	# ax.set_xticks(np.arange(1,11))
	# ax.set_yticks(np.arange(12,22))
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# ax.set_yticklabels(np.arange(12,22))
	# ax.set_zlim(-1,1.5)
	# ax.plot_surface(x,y,z)
	# plt.show()

	# print(len(agent.values))
	# print(sorted(agent.values.items(), key=lambda k: agent.values[k[0]]))

	# valsp = np.zeros(20)
	# print('17 vals')
	# for i in range(1,11):
	# 	print('Dealer upcard: ', i)
	# 	valsp[i] = agent.values[(17, i, False), 1]
	# print(valsp)

	# val_hit_17 = [-0.75325902, -0.68544511, -0.68846589, -0.67651544, -0.69791068, -0.67434516, -0.68209819, -0.66635285, -0.66854884, -0.69916342]
	# val_stick_17 = [-0.63062319, -0.14728525, -0.09830716, -0.07287693, -0.0489353, 0, -0.1028798,  -0.39850686, -0.43407751, -0.46759447]

	# figure = plt.figure()
	# ax = plt.axes()
	# plt.scatter(np.arange(1,11), val_stick_17, color='green', label = 'Always STICK on 17')
	# plt.scatter(np.arange(1,11), val_hit_17, color='blue', label = 'Always HIT on 17')
	# ax.set_xticks(np.arange(1,11))
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# plt.legend()
	# plt.title('Value with 17 vs. dealer upcard')
	# plt.xlabel('Dealer upcard')
	# plt.ylabel('Value')
	# plt.show()

	# figure = plt.figure()
	# ax = plt.axes(projection = '3d')
	# x = np.arange(1,11)
	# y = np.arange(12, 22)
	# plt.xlabel('Dealer showing')
	# plt.ylabel('Player sum')
	# plt.title('Values with no usable ace 500,000 episodes HIT on >= 20')
	# z = np.zeros((10, 10))
	# for i in range(1,11):
	# 	for j in range(12,22):
	# 		if j>=20: 
	# 			h = 0
	# 		else:
	# 			h = 1
	# 		z[j-12, i-1] = agent.values[(j, i, False), h] #our card, dealer card, valid Ace
	# print('z', z)
	# x,y = np.meshgrid(x,y)
	# ax.set_xticks(np.arange(1,11))
	# ax.set_yticks(np.arange(12,22))
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# ax.set_yticklabels(np.arange(12,22))
	# ax.set_zlim(-1,1)
	# ax.plot_surface(x,y,z)
	# plt.show()

	# figure = plt.figure()
	# ax = plt.axes(projection = '3d')
	# x = np.arange(1,11)
	# y = np.arange(12, 22)
	# plt.xlabel('Dealer showing')
	# plt.ylabel('Player sum')
	# plt.title('Value with usable ace 500,000 episodes HIT on >= 20')
	# z = np.zeros((10, 10))
	# for i in range(1,11):
	# 	for j in range(12,22):
	# 		if j>=20: 
	# 			h = 0
	# 		else:
	# 			h = 1
	# 		z[j-12, i-1] = agent.values[(j, i, True), h] #our card, dealer card, valid Ace
	# print('z', z)
	# x,y = np.meshgrid(x,y)
	# ax.set_xticks(np.arange(1,11))
	# ax.set_yticks(np.arange(12,22))
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# ax.set_yticklabels(np.arange(12,22))
	# ax.set_zlim(-1,1)
	# ax.plot_surface(x,y,z)
	# plt.show()
	

	# figure = plt.figure()
	# ax = plt.axes(projection = '3d')
	# x = np.arange(1,11)
	# y = np.arange(12, 22)
	# plt.xlabel('Dealer showing')
	# plt.ylabel('Player sum')
	# plt.title('Value with usable ace')
	# z = np.zeros((10, 10))
	# for i in range(1,11):
	# 	for j in range(12,22):
	# 		if j >= 20:
	# 			h = 1
	# 		else: 
	# 			h = 0
	# 		z[j-12, i-1] = agent.values[(j, i, True), h] #our card, dealer card, valid Ace
	# x,y = np.meshgrid(x,y)
	# ax.set_xticks(np.arange(1,11))
	# ax.set_yticks(np.arange(12,22))
	# ax.set_xticklabels(['A', 2, 3, 4, 5, 6, 7, 8, 9, 'T'])
	# ax.set_yticklabels(np.arange(12,22))
	# ax.set_zlim(-1,1)
	# ax.plot_surface(x,y,z)
	# plt.show()

#this part is for when you can see both dealer cards

	s = collections.defaultdict(str)
	pi = np.zeros((14, 16))
	for dealer in range(5,21):
		# print('dealer', dealer)
		for player in range(8,22):
			best_action = np.argmax(agent.policy[(player,dealer,False)])
			#print('policy', agent.policy[(player,dealer,False)])
			pi[player-8, dealer-5] = best_action #our card, dealer card, valid Ace
			if best_action == 0:
				s[player-8, dealer-5] = 'S'
			elif best_action == 1:
				s[player-8, dealer-5] = 'H'
			else:
				s[player-8, dealer-5] = 'D'
			#print('Dealer showing: {}, Player total: {}, Best action: {})\n'.format(dealer, player, best_action))

	print(pi)
	pi = np.flipud(pi)
	fig, ax = plt.subplots()
	im = ax.imshow(pi, extent=[4.5, 20.5, 7.5, 21.5])
	ax.set_xticks(np.arange(5,21))
	ax.set_yticks(np.arange(8,22))
	ax.set_title('Optimal strategy no usable Ace')
	ax.set_xlabel('Dealer sum')
	ax.set_ylabel('Player sum')
	for i in range(5,21):
		for j in range(8,22):
			text = ax.text(i, j, s[j-8, i-5], ha="center", va="center")
	ax.set_xticklabels(np.arange(5,21))
	ax.set_yticklabels(np.arange(8,22))

	plt.show()

	s = collections.defaultdict(str)
	pi = np.zeros((10, 16))
	for dealer in range(5,21):
		# print('dealer', dealer)
		for player in range(12,22):
			best_action = np.argmax(agent.policy[(player,dealer,True)])
			pi[player-12, dealer-5] = best_action #our card, dealer card, valid Ace
			if best_action == 0:
				s[player-12, dealer-5] = 'S'
			elif best_action == 1:
				s[player-12, dealer-5] = 'H'
			else:
				s[player-12, dealer-5] = 'D'
			#print('Dealer showing: {}, Player total: {}, Best action: {})\n'.format(dealer, player, best_action))

	print(pi)
	pi = np.flipud(pi)
	fig, ax = plt.subplots()
	im = ax.imshow(pi, extent=[4.5, 20.5, 11.5, 21.5])
	ax.set_xticks(np.arange(5,21))
	ax.set_yticks(np.arange(12,22))
	ax.set_title('Optimal strategy usable Ace')
	ax.set_xlabel('Dealer sum')
	ax.set_ylabel('Player sum')
	for i in range(5,21):
		for j in range(12,22):
			text = ax.text(i, j, s[j-12, i-5], ha="center", va="center")
	ax.set_xticklabels(np.arange(5,21))
	ax.set_yticklabels(np.arange(12,22))

	plt.show()


	figure = plt.figure()
	ax = plt.axes(projection = '3d')
	x = np.arange(5,21)
	y = np.arange(8, 22)
	plt.xlabel('Dealer sum')
	plt.ylabel('Player sum')
	plt.title('Optimal value with no usable Ace')
	z = np.zeros((14, 16))
	for i in range(5,21):
		for j in range(8,22):
			z[j-8, i-5] = agent.values[(j, i, False), np.argmax(agent.policy[(j,i,False)])] #our card, dealer card, valid Ace
	x,y = np.meshgrid(x,y)
	ax.set_xticks(np.arange(5,21))
	ax.set_yticks(np.arange(8,22))
	ax.set_xticklabels(np.arange(5,21))
	ax.set_yticklabels(np.arange(8,22))
	ax.set_zlim(-1,1)
	ax.plot_surface(x,y,z)
	plt.show()


	figure = plt.figure()
	ax = plt.axes(projection = '3d')
	x = np.arange(5,21)
	y = np.arange(12, 22)
	plt.xlabel('Dealer showing')
	plt.ylabel('Player sum')
	plt.title('Optimal value with usable Ace')
	z = np.zeros((10, 16))
	for i in range(5,21):
		for j in range(12,22):
			z[j-12, i-5] = agent.values[(j, i, True), np.argmax(agent.policy[(j,i,True)])] #our card, dealer card, valid Ace
	x,y = np.meshgrid(x,y)
	ax.set_xticks(np.arange(5,21))
	ax.set_yticks(np.arange(12,22))
	ax.set_xticklabels(np.arange(5,21))
	ax.set_yticklabels(np.arange(12,22))
	ax.set_zlim(-1,1.5)
	ax.plot_surface(x,y,z)
	plt.show()