from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np


def main():
    env = StarCraft2Env(map_name="3m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    #print(n_actions)
    #print(n_agents)
    n_episodes = 100

    for e in range(n_episodes):
        env.reset()
        #print("-------------get_avail_agent_actions------------------")
        #print(env.get_avail_agent_actions(0))
        #print("-------------get_env_info------------------")
        #print(env.get_env_info())
        #print("-------------get_obs------------------")
        #print(env.get_obs())
        print("-------------get_state------------------")
        print(env.get_state())
        #print("-------------get_obs_agent------------------")
        #print(env.get_obs_agent(0))
        #print("-------------get_obs_size------------------")
        #print(env.get_obs_size())
        #print("-------------get_state_size------------------")
        #print(env.get_state_size())
        print("-------------get_avail_actions------------------")
        print(env.get_avail_actions())
        print("-------------get_total_actions------------------")
        print(env.get_total_actions())

        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            #print(actions)
            #print("-----------------")
            reward, terminated, env_info = env.step(actions)
            print(terminated)
            print(reward)
            print(env_info)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
