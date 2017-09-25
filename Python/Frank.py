from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plt
#import scipy.misc
import os
#from keras.activations import relu
#%matplotlib inline

import time

env = gym.make("CartPole-v0")
print("action space: " + str(env.action_space.n))
#env = gym.make("SpaceInvaders" + "NoFrameskip-v4")
#env = gym.make("SpaceInvaders-v0")

class Qnetwork():
	def __init__(self,h_size):
		#The network recieves a frame from the game, flattened into an array.
		#It then resizes it and processes it through four convolutional layers.
		self.flat_input =  tf.placeholder(shape=[None,INPUT_SIZE],dtype=tf.float32)
# 		if ENABLE_CONVOLUTION:
# 			self.imageIn = tf.reshape(self.flat_input,shape=[-1,210,160,3])
# 			#self.imageIn = tf.placeholder(shape = [None, 210, 160, 3], dtype = tf.float32) #None here is batch size - number of images to process?
# 			print("imageIn: " + str(self.imageIn.get_shape()))
# 			self.conv1 = slim.conv2d( \
# 				inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
# 			print("conv1: " + str(self.conv1.get_shape()))
# 			self.conv2 = slim.conv2d( \
# 				inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
# 			print("conv2: " + str(self.conv2.get_shape()))
# 			self.conv3 = slim.conv2d( \
# 				inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None)
# 				#inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
# 	
# 			print("conv3: " + str(self.conv3.get_shape()))
# 			self.conv4 = slim.conv2d( \
# 				inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[6,2],padding='VALID', biases_initializer=None)
# 			print("conv4: " + str(self.conv4.get_shape()))	
# 			#We take the output from the final convolutional layer and split it into separate advantage and value streams.
# 			#self.streamAC,self.streamVC = tf.split(self.conv4,2,3) #Should be 1 x 1 x 1 256
# 			self.flat_features = self.conv4
# 		else:
			
		self.flat_features = self.flat_input
		
		self.streamAC = self.flat_features
		self.streamVC = self.flat_features
		
		#print("streamAC", self.streamAC.get_shape())
		self.streamA = slim.flatten(self.streamAC)
		self.streamV = slim.flatten(self.streamVC)
		xavier_init = tf.contrib.layers.xavier_initializer() #Initializes weights proportionate to input layer
		
		#Output
		#print("action space: " + str(env.action_space.n) + ", " + str(type(env.action_space.n)))
		#self.AW = tf.Variable(xavier_init([h_size,env.action_space.n])) #Action weights
		#self.VW = tf.Variable(xavier_init([h_size,1])) #Value weights
		
		#FC layers
		self.advantage_fc1 = slim.fully_connected(self.streamA, h_size, weights_initializer=xavier_init, activation_fn=tf.nn.relu, biases_initializer= xavier_init)
		self.advantage_fc2 = slim.fully_connected(self.advantage_fc1, h_size, weights_initializer=xavier_init, activation_fn=tf.nn.relu, biases_initializer= xavier_init)
		
		self.value_fc1 = slim.fully_connected(self.streamV, h_size, weights_initializer=xavier_init, activation_fn=tf.nn.relu, biases_initializer= xavier_init)
		self.value_fc2 = slim.fully_connected(self.value_fc1, h_size, weights_initializer=xavier_init, activation_fn=tf.nn.relu, biases_initializer= xavier_init)
		
		#Output
		self.Advantage = slim.fully_connected(self.advantage_fc2, env.action_space.n, weights_initializer=xavier_init, activation_fn=tf.nn.softmax, biases_initializer= xavier_init)
		self.Value = slim.fully_connected(self.value_fc2, 1, weights_initializer=xavier_init, activation_fn= None, biases_initializer= xavier_init)

		#self.Advantage = tf.matmul(self.streamA,self.AW) # 1 x 256 (H) * 256(H) x 6 (action_size)
		#self.Value = tf.matmul(self.streamV,self.VW)
		
		#Then combine them together to get our final Q-values.
		self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
		self.predict = tf.argmax(self.Qout,1)
		
		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,env.action_space.n,dtype=tf.float32)
		
		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
		
		self.td_error = tf.square(self.targetQ - self.Q)
		self.loss = tf.reduce_mean(self.td_error)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.updateModel = self.trainer.minimize(self.loss)
		
class experience_buffer():
	def __init__(self, buffer_size = 50000):
		self.buffer = []
		self.buffer_size = buffer_size
	
	def add(self,experience):
		if len(self.buffer) + len(experience) >= self.buffer_size:
			self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
		self.buffer.extend(experience)
			
	def sample(self,size):
		return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
	
def processState(states, size):
	return np.reshape(states,size)

def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars//2]):
		op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)
		
batch_size = 32 #How many experiences to use for each training step.
update_freq = 100 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 10000 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./SpaceInvaders" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

#New constants
INPUT_SIZE = 4 #State size (Use 100800 for Atari)
SAVE_FREQUENCY = 10 #How many episodes to wait between saves
LOG_FREQUENCY = 100
ENABLE_CONVOLUTION = False

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
	os.makedirs(path)

with tf.Session() as sess:
	
	sess.run(init)
	if load_model == True:
		print('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	for i in range(num_episodes):
		episodeBuffer = experience_buffer()
		#Reset environment and get first new observation
		s = env.reset()
		s = processState(s, INPUT_SIZE)
		d = False
		rAll = 0
		j = 0
		#The Q-Network
		while j < max_epLength:
			#env.render()
			j+=1
			
			#Choose an action by greedily (with e chance of random action) from the Q-network
			if np.random.rand(1) < e or total_steps < pre_train_steps:
				a = env.action_space.sample()
			else:
				time_before_feed_forward = time.time()
				a = sess.run(mainQN.predict,feed_dict={mainQN.flat_input:[s]})[0]
				
				#time_after_feed_forward = time.time()
				#feed_forward_elapsed_time = time_after_feed_forward - time_before_feed_forward
				#print("feed forward time: ", feed_forward_elapsed_time)
			
			#print("advantage: " + str(sess.run(mainQN.Advantage, feed_dict = {mainQN.flat_input:[s]}))) 
			#print("chosen action: " + str(a))
			s1,r,d, info = env.step(a) #Last arg is info, containing lives
			#print("steps: ", env._elapsed_steps)
			#print("info", info)
			
			#End epoch after single death
			#lives = info['ale.lives']
			#if lives < 3:
			#	d = True
			
			s1 = processState(s1, INPUT_SIZE)
			total_steps += 1
			episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
			
			if total_steps > pre_train_steps:
				if e > endE:
					e -= stepDrop
				
				if total_steps % (update_freq) == 0:
					trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
					#Below we perform the Double-DQN update to the target Q-values
					Q1 = sess.run(mainQN.predict,feed_dict={mainQN.flat_input:np.vstack(trainBatch[:,3])}) #Column 3 is state
					Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.flat_input:np.vstack(trainBatch[:,3])})
					end_multiplier = -(trainBatch[:,4] - 1) #Column 4 is done
					doubleQ = Q2[range(batch_size),Q1]
					targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
					#Update the network with our target values.
					_ = sess.run(mainQN.updateModel, \
						feed_dict={mainQN.flat_input:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
					
					backprop_start_time = time.time()
					updateTarget(targetOps,sess) #Update the target network toward the primary network.
					backprop_end_time = time.time()
					backprop_elapsed_time = backprop_end_time - backprop_start_time
					#print("backprop elapsed time: ", backprop_elapsed_time)
			rAll += r
			s = s1
			
			if d == True:
				#print("episode #" + str(i) + " - " + "reward: " + str(rAll) + ", steps: " + str(j))
				break
		
		myBuffer.add(episodeBuffer.buffer)
		jList.append(j)
		rList.append(rAll)
		#Periodically save the model. 
		if i % SAVE_FREQUENCY == 0:
			saver.save(sess,path+'/model-'+str(i)+'.ckpt')
			#print("Saved Model")
		if len(rList) % LOG_FREQUENCY == 0:
			#print("FC 1 weights" + str(tf.get_variable('fully_connected/weights')))
			print("Synapse weights: " + str(sess.run(tf.all_variables())[0]))
			print("Mean reward at " + str(i) + ": " + str(np.mean(rList[-SAVE_FREQUENCY:])))
	saver.save(sess,path+'/model-'+str(i)+'.ckpt')
#print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")