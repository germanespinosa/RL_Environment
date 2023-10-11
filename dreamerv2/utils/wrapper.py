# import minatar
import gym
import numpy as np
from gym_bins_env import Environment
class CustomGymEnv(gym.Env):
    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env = env = Environment(e=2, has_predator="True", predator_speed=0.2, max_step=300)
        self.action_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

    def close(self):
        pass

    
## Sticky action
# class ActionRepeat(gym.Wrapper):
#     def __init__(self, env, repeat=1):
#         super(ActionRepeat, self).__init__(env)
#         self.repeat = repeat
#
#     def step(self, action):
#         done = False
#         total_reward = 0
#         current_step = 0
#         while current_step < self.repeat and not done:
#             obs, reward, done, truncated, info = self.env.step(action)
#             total_reward += reward
#             current_step += 1
#         return obs, total_reward, done, truncated, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0
    
    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, truncated, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()

class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()
    
    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference
