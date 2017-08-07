






class Agent() :
	def __init__ (self, learning_rate, nodes_per_layer, hidden_layer_count):
		#Create placeholders and operations
		self.chosen_action = ""; #Controller output

episode_count = 2000

backprop_frequency = 5

for i in range(episode_count):
#For each episode
	running_reward = 0.

	#For each frame
	
		#Get window pixels
		
		#Forward feed through convolutional layers to get input features

		#Forward feed input features through fully connected hidden layers
	
		#Choose between network output and random action

	if i != 0 and i % backprop_frequency == 0:
		#Backpropagate reward
