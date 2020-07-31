import logging
import requests
from typing import Any, Dict
from cartpole import CartPole

class BonsaiAgent(object):
    """ The agent that gets the action from the trained brain exported as docker image and started locally
    """
    def act(self, state)->Dict[str, Any]:
        action = self.predict(state)
        action["command"] = int(action["command"])
        return action

    def predict(self, state):
        url = "http://localhost:5000/v1/prediction"

        response = requests.get(url, json=state)
        action = response.json()

        return action

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, cartpole:CartPole):
        self.cartpole = cartpole

    def act(self, state):
        return cartpole.gym_to_action(cartpole._env.action_space.sample())


if __name__ == '__main__':
    logging.basicConfig()
    log = logging.getLogger("cartpole")
    log.setLevel(level='DEBUG')

    # we will use our environment (wrapper of OpenAI env)
    cartpole = CartPole()


    # specify which agent you want to use, 
    # BonsaiAgent that uses trained Brain or
    # RandomAgent that randomly selects next action
    agent = RandomAgent(cartpole._env.action_space)

    episode_count = 100
    reward = 0
    done = False

    try:
        for i in range(episode_count):

            cartpole.episode_start()
            state = cartpole.get_state()

            while True:

                action = agent.act(state)
                print(action)
                cartpole.episode_step(action)
                state = cartpole.get_state()
                
                if cartpole.halted():
                    break

            cartpole.episode_finish("")
    except KeyboardInterrupt:
        print("Stopped")