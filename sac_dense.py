from gym_env import Environment
#from gym_env_6obs import Environment
#from gym_env_predator import Environment
from cellworld import *
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common import env_checker
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq: int, stop_reward: float, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.stop_reward = stop_reward

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Assume the environment uses a Gym interface
            # and calculate the mean reward over last 100 episodes
            x = self.model.get_env().get_attr('episode_reward_history')[0][-100:]
            mean_reward = round(np.mean(x), 1)
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {mean_reward} - Last 5 reward: {x[-5:]}")
            # Stop training if mean reward exceeds the set threshold
            if mean_reward >= self.stop_reward:
                return False
        return True


## check the env
def check():
    e = Environment("21_05", freq=100, has_predator=False)
    env_checker.check_env(e, warn=True)

def random_policy():
    done = False
    env = Environment(freq=100, has_predator=False)
    env.reset()
    #while not done:
    for i in range(5000):
        obs, reward, done, _, _ = env.step([1, 0])
        if i%20==0:
            env.reset()
        #obs, reward, done, _, _ = wrapped_env.step(wrapped_env.action_space.sample())
        env.show()
    env.close()


def PPO_train():
    env = Environment("16_05", freq=100, has_predator=False, max_step=200)
    env.reset()
    # PPO has no action noise, but We can apply that in the step method. Also, ent_coef ent_coef=0.01, #seems helpful refers to the policy's entropy which might help exploration
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-4,
                policy_kwargs={"net_arch": [128, 256, 128]},
                )

    callback = EarlyStoppingCallback(check_freq=1000, stop_reward=3000)
    model.learn(total_timesteps=200000, log_interval=10, callback=callback)
    plot(env.episode_reward_history, name="16_05_reward-100_PPO")
    model.save("1605PPO_model")
    env.close()
def SAC_train_random_env():
    # we train it on 0-19, and test it on 20-30
    env = Environment(e=5, has_predator=False, max_step=300, env_type="train")
    sigma = 0.1 # Adjust this value to increase the noise level
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=sigma * np.ones(env.action_space.shape))
    model = SAC("MlpPolicy", # batchsize
                env,
                verbose=1,
                batch_size = 512,
                learning_rate= 3e-4,
                train_freq=(1, "step"),
                buffer_size=2500000,
                replay_buffer_class=ReplayBuffer,
                action_noise=action_noise,  # noise level
                policy_kwargs={"net_arch": [128, 256, 128]} # the network
                )
    callback = EarlyStoppingCallback(check_freq=3000, stop_reward=300)
    model.learn(total_timesteps=1500000, log_interval=10, callback=callback)
    plot(env.episode_reward_history, name="50_SAC")
    model.save("50_SAC")
    env.close()

def SAC_train():
    env = Environment("17_09", freq=100, has_predator=False, max_step=300)
    env.reset()
    sigma = 0.1 # Adjust this value to increase the noise level
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=sigma * np.ones(env.action_space.shape))
    model = SAC("MlpPolicy", # batchsize
                env,
                verbose=1,
                batch_size = 512,
                learning_rate= 3e-4,
                train_freq=(1, "step"),
                buffer_size=2000000,
                replay_buffer_class=ReplayBuffer,
                action_noise=action_noise,  # noise level
                policy_kwargs={"net_arch": [128, 256, 128]} # the network
                )
    model = SAC.load("1709", env)
    callback = EarlyStoppingCallback(check_freq=5000, stop_reward=300)
    model.learn(total_timesteps=1000000, log_interval=20, callback=callback)
    plot(env.episode_reward_history, name="1709_SAC_reward")
    model.save("1709")
    env.close()

def plot(data, name ="result"):
    import matplotlib.pyplot as plt
    import pandas as pd
    data_series = pd.Series(data)
    smooth_data = data_series.rolling(window=100).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Reward')
    plt.plot(smooth_data, color='red', label='Smoothed Reward')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.title('Reward History', fontsize=18)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name, dpi=300)

def result_visualization():
    # from stable_baselines3.common.evaluation import evaluate_policy
    # env_type="train"
    env = Environment(freq=100, has_predator=False, max_step=250, env_type="train")
    #loaded_model = SAC.load("1601_SAC_pre-tun.zip")
    #model = SAC("MlpPolicy", verbose=1, env = env, seed=123, learning_rate=0.0003)
    loaded_model = SAC.load("50_SAC_pre-tun-1")
    #loaded_model = PPO.load("16_04_predator_PPO.zip")
    #model.set_parameters("stable_sac")
    obs, _ = env.reset()
    #obs, _ = wrapped_env.reset()
    done = False
    tr = False
    while not (done or tr):
        action, _states = loaded_model.predict(obs, deterministic=True)
        # action, _states = model.predict(obs)
        obs, reward, done, tr, _ = env.step(action)
        env.show()
        if done:
            # obs,_ = wrapped_env.reset()
            env.close()
    env.close()

def evaluate():
    np.random.seed(123)
    env = Environment("16_05", freq=100, has_predator=False)
    #model = SAC("MlpPolicy", env, verbose=1, seed=123, learning_rate=0.0003)
    loaded_model = SAC.load("SAC_16_06.zip")
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=50)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

def check_prediction():
    env = Environment("16_05", freq=100, has_predator=False)
    #model = SAC("MlpPolicy", env, verbose=1, seed=123, learning_rate=0.0003)
    #model.set_parameters("stable_sac")
    loaded_model = SAC.load("stable_sac")
    obs, _ = env.reset()
    # Check prediction before saving
    print("pre saved", loaded_model.predict(obs, deterministic=True))


if __name__ == "__main__":
    #check()
    SAC_train_random_env() # SAC_train()
    # result_visualization()
    # random_policy()
    #evaluate()
    #check_prediction()
    #PPO_train()