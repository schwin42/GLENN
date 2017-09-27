import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


print("program started")

class Agent() :
	def __init__ (self, learning_rate, nodes_per_layer, hidden_layer_count):
		#Create placeholders and operations
		self.state = tf.placeholder(shape = [None, INPUT_SIZE], dtype = tf.float32)
		xavier_init = tf.contrib.layers.xavier_initializer()
		#print("state rank" + str(tf.rank(state)))
		#print("state shape" + str(tf.shape(state)))

		#Advantage network
		self._advantage_fc_layers = []
		_layer_input = self.state
		for i in range(hidden_layer_count):
			layer = slim.fully_connected(_layer_input, nodes_per_layer, weights_initializer = xavier_init, biases_initializer = xavier_init, activation_fn = tf.nn.relu) 
			self._advantage_fc_layers.append(layer)
			_layer_input = layer
		
		#Value network
		# self.value_fc_layers = []
		# _layer_input = state
		# for i in range(hidden_layer_count):
		# 	layer = slim.fully_connected(state, nodes_per_layer, weights_initializer = xavier_init, bias_initializer = tf.xavier_init, activation_fn = tf.nn.relu) 
		# 	self.value_fc_layers.append(layer)
		# 	layer_input = layer

		#Output
		self.advantage_prediction = slim.fully_connected(self._advantage_fc_layers[-1], env.action_space.n, weights_initializer = xavier_init, biases_initializer = xavier_init, activation_fn = tf.nn.softmax) #Should this really be softmax or should this represent expected reward?
		#self value_output_layer = slim.fully_connected(value_fc_layers[-1], 1, weights_initializer = xavier_init, bias_initializer = tf.xavier_init)
		self.chosen_action = tf.argmax(self.advantage_prediction, 1)[0]



#Constants
epoch_count = 50
backprop_frequency = 5
max_epoch_length = 200

#Environment constants
ENVIRONMENT_NAME = "CartPole-v0"
INPUT_SIZE = 4

#Initialize environment
env = gym.make(ENVIRONMENT_NAME)

#Initalize agent
agent = Agent(0.0001, 128, 2)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	total_reward_by_episode = []
	for i in range(epoch_count):
	#For each episode
		#print("Starting episode: " + str(i))
		running_reward = 0.
		state = env.reset()
		is_done = False

		#For each step
		for j in range(max_epoch_length):
			#print("Beginning of step: " + str(j))
			env.render()
		
			#Forward feed input features through fully connected hidden layers
			print("advantage: " + str(sess.run(agent.advantage_prediction, feed_dict = {agent.state: [state]})))
			print("State: " + str(state))
			action = sess.run(agent.chosen_action, feed_dict = {agent.state: [state]})

			#TODO Choose between network output and random action
			#print("ad prob" + str(advantage_probabilities))
			
			print("action: " + str(action))
			
			observation, reward, is_done, info = env.step(action)
			#Choose random action - observation, reward, done, info = env.step(env.action_space.sample()) #Choose random action
			running_reward += reward

			if is_done == True:
				total_reward_by_episode.append(running_reward)
				print("Reward for episode " + str(i) + ": " + str(running_reward))
				if(i != 0 and i % 10 == 0):
					print("Mean reward for last ten episodes: " + str(np.mean(total_reward_by_episode)))
				break


		#if i != 0 and i % backprop_frequency == 0:
			#print("backprop")
			#Backpropagate reward






print("program terminated")
