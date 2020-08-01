import gym
from gym import logger
import requests


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('CarRacing-v0')

    env.seed(0)

    # specify which agent you want to use,
    # BonsaiAgent that uses trained Brain or
    # RandomAgent that randomly selects next action
    agent = RandomAgent(env.action_space)

    env.render()
    env.reset()

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            print(ob)
            env.render()
            if done:
                break
    env.close()
