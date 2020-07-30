import gym
from gym import logger
import requests


class BonsaiAgent(object):
    """ The agent that gets the action from the trained brain exported as docker image and started locally
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        action = self.predict(observation[0], observation[1], observation[2], observation[3])
        return action["command"]

    def predict(self,
                cart_position: float,
                cart_velocity: float,
                pole_angle: float,
                pole_angular_velocity : float) -> dict:
        url = "http://localhost:5000/v1/prediction"
        state = {
            "cart_position": cart_position,
            "cart_velocity": cart_velocity,
            "pole_angle": pole_angle,
            "pole_angular_velocity": pole_angular_velocity
        }

        response = requests.get(url, json=state)
        action = response.json()

        return action


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

    env = gym.make('CartPole-v0')

    env.seed(20)

    # specify which agent you want to use, 
    # BonsaiAgent that uses trained Brain or
    # RandomAgent that randomly selects next action
    agent = BonsaiAgent(env.action_space)


    env.render()
    env.reset()

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        episode_step = 0
        #print(ob)
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(int(action))

            episode_step += 1
            
            env.render()
            if done:
                print("done")
                break
 
    env.close()
