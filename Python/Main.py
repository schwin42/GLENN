import gym
import numpy as np
import tensorflow as tf


print("program started")

class Agent() :
	def __init__ (self, learning_rate, nodes_per_layer, hidden_layer_count):
		#Create placeholders and operations
		self.chosen_action = "" #Controller output

#Constants
epoch_count = 50
backprop_frequency = 5
max_epoch_length = 200

#Initialize environment
env = gym.make("CartPole-v0")

total_reward_by_episode = []
for i in range(epoch_count):
#For each episode
	#print("Starting episode: " + str(i))
	running_reward = 0.
	env.reset()

	#For each step
	for j in range(max_epoch_length):
		#print("Beginning of step: " + str(j))
		env.render()
		#Forward feed input features through fully connected hidden layers
	
		#Choose between network output and random action

		observation, reward, done, info = env.step(env.action_space.sample()) #Choose random action
		running_reward += reward

		if done == True:
			total_reward_by_episode.append(running_reward)
			print("Reward for episode " + str(i) + ": " + str(running_reward))
			if(i != 0 and i % 10 == 0):
				print("Mean reward for last ten episodes: " + str(np.mean(total_reward_by_episode)))
			break


	#if i != 0 and i % backprop_frequency == 0:
		#print("backprop")
		#Backpropagate reward






print("program terminated")
