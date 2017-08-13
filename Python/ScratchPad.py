import gym

env = gym.make('SpaceInvaders-v0')
# env = gym.make('Asteroids-v0')
env.reset()


for i in range(100000):
	env.render()
	observation, reward, done, info = env.step(env.action_space.sample())
	print("info: ", info)
	lives = info['ale.lives']
	print("lives: ", lives)
	print("action space", env.action_space)
	print("observation shape: " + str(len(observation)) + ", " + str(len(observation[0])) + ", " + str(len(observation[0][0])))
