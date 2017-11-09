import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from Model import ExperienceBuffer
#from turtle import done

print("program started")

class Agent() :
	def __init__ (self, learning_rate, nodes_per_layer, hidden_layer_count):
		
		#Create experience buffer
		self.experience_buffer = ExperienceBuffer()
		
		#Create placeholders and operations
		self.state = tf.placeholder(shape = [None, INPUT_SIZE], dtype = tf.float32)
		xavier_init = tf.contrib.layers.xavier_initializer()
		#print("state rank" + str(tf.rank(state)))
		#print("state shape" + str(tf.shape(state)))

		#Advantage network
		self._advantage_fc_layers = []
		_layer_input = self.state
		for _ in range(hidden_layer_count):
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
		self.forwardProp_quality = slim.fully_connected(self._advantage_fc_layers[-1], env.action_space.n, weights_initializer = xavier_init, biases_initializer = xavier_init, activation_fn = tf.nn.softmax) #Should this really be softmax or should this represent expected reward?
		#TODO Dueling functionality - split network into value and advantage streams
		#self value_output_layer = slim.fully_connected(value_fc_layers[-1], 1, weights_initializer = xavier_init, bias_initializer = tf.xavier_init)
		self.chosen_action = tf.argmax(self.forwardProp_quality, 1)[0]
		
		#Update
		
		#Calculate quality error
		
		#Zero out error for non-chosen actions
		
		#Use adam optimizer to minimize loss
		
		self.backProp_experience
		#self.backProp_chosen_action

def select_boltzmann_action(action_qualities):
	lower_bound = 0.
	random_number = np.random.rand(1)
	#print("random: ", random_number)
	#print("array: ", action_qualities)

	for index, value in enumerate(action_qualities):
#		print("action: ", action_qualities[index])
		#print("index: ", str(index))
		if random_number >= lower_bound and random_number <= value + lower_bound:
			#print("returning ", index)
			return index
		else:
			lower_bound += value
			if(index >= len(action_qualities) - 1):
				raise ValueError("Assumption didn't hold for select_boltzmann_action")

#Constants
epoch_count = 50
#backprop_frequency = 5
max_epoch_length = 200
batch_size = 100 #How many experiences to backprop per update

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
			#print("qualities: " + str(sess.run(agent.forwardProp_quality, feed_dict = {agent.state: [state]})))
# 			print("State: " + str(state))
			#action = sess.run(agent.chosen_action, feed_dict = {agent.state: [state]})
			action_qualities = sess.run(agent.forwardProp_quality, feed_dict = {agent.state: [state]})
			
			action = select_boltzmann_action(action_qualities[0])
			
			#print("action: " + str(action))
			
			observation, reward, is_done, info = env.step(action)
			#Choose random action - observation, reward, done, info = env.step(env.action_space.sample()) #Choose random action
			running_reward += reward
			
			#Add experience to buffer
			experience_array = np.array([state, action, reward, observation, is_done])
			experience = np.reshape(experience_array, [1,5])
			agent.experience_buffer.add(experience)
			
			#episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.


			if is_done == True:
				total_reward_by_episode.append(running_reward)
				print("Reward for episode " + str(i) + ": " + str(running_reward))
				if(i != 0 and i % 10 == 0):
					print("Mean reward for last ten episodes: " + str(np.mean(total_reward_by_episode)))
					
				#Perform backprop at the end of every episode
				training_batch = agent.experience_buffer.sample(batch_size) #Experience features x number of experiences
				
				
				break #Episode is over, bail out of step iterator


		#if i != 0 and i % backprop_frequency == 0:
			#print("backprop")
			#Backpropagate reward






print("program exited gracefully")
