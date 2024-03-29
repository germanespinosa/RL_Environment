import copy
from prey_env import gymnasium_Environment_C as Environment
from cellworld import *
import numpy as np
from stable_baselines3 import PPO
#from stable_baselines3 import SAC
from sac.sacwithicm import SAC
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
    env = Environment(freq=100, max_step=500, e=1, predator_speed=0.1, env_random=True, has_predator=True)
    # env = Environment(freq=100, has_predator=True, max_step=500, e=9, predator_speed=0.1)
    env.reset()
    #while not done:
    for i in range(50000):
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        if i % 5000==0:
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
    # we train it on 0-10, and test it on 20-30
    env = Environment(e=2, has_predator=True, max_step=300, env_type="train", predator_speed=0.2)
    sigma = 0.1 # Adjust this value to increase the noise level
    # action_noise_mean_scale = 0.24 * 0.3
    # action_noise = NormalActionNoise(mean=action_noise_mean_scale * np.ones(env.action_space.shape),
    #                                  sigma=sigma * np.ones(env.action_space.shape))
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=sigma * np.ones(env.action_space.shape))
    model = SAC("MlpPolicy", # batchsize
                env,
                verbose=1,
                batch_size = 512,
                learning_rate= 3e-4,
                train_freq=(1, "step"),
                buffer_size=1500000,
                replay_buffer_class=ReplayBuffer,
                action_noise=action_noise,  # noise level
                policy_kwargs={"net_arch": [128, 256, 128]} # the network
                )
    # model = SAC.load("e2r10", env)
    # callback = EarlyStoppingCallback(check_freq=3000, stop_reward=300)
    model.learn(total_timesteps=500000, log_interval=10)
    plot(env.episode_reward_history, name="e2r10")
    model.save("e2r10")
    env.close()






def TD3_train_random_env():
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise

    # we train it on 0-19, and test it on 20-30
    env = Environment(e=2, has_predator=True, max_step=300, env_type="train", predator_speed=0.2)
    sigma = 0.1
    action_noise_mean_scale = 0.24 * 0.3
    action_noise = NormalActionNoise(mean=action_noise_mean_scale * np.ones(env.action_space.shape),
                                     sigma=sigma * np.ones(env.action_space.shape))

    model = TD3("MlpPolicy",
                env,
                verbose=1,
                batch_size=512,
                learning_rate=3e-4,
                buffer_size=1500000,
                train_freq=1000,  # You might want to adjust this value for TD3
                action_noise=action_noise,
                policy_kwargs={"net_arch": [128, 256, 128]},
                policy_delay=2,  # This is specific to TD3
                learning_starts=10000  # TD3 typically waits for a certain number of steps before training starts
                )

    model.learn(total_timesteps=500000, log_interval=10)
    # Assuming you have a plot function similar to before
    plot(env.episode_reward_history, name="e2r50TD3")
    model.save("e2r50TD3")
    env.close()

def random_env_training():
    # initilize env
    env = Environment(e=3, has_predator=False, max_step=300, env_type="train", predator_speed=0.2)
    sigma = 0.1  # Adjust this value to increase the noise level
    action_noise_mean_scale = 0.24 * 0.3
    action_noise = NormalActionNoise(mean=action_noise_mean_scale * np.ones(env.action_space.shape),
                                     sigma=sigma * np.ones(env.action_space.shape))
    model = SAC("MlpPolicy",
                env,
                verbose=1,
                batch_size=512,
                learning_rate=3e-4,
                train_freq=(1, "step"),
                buffer_size=1500000,
                replay_buffer_class=ReplayBuffer,
                action_noise=action_noise,  # noise level
                policy_kwargs={"net_arch": [128, 256, 128]}  # the network
                )
    # first train
    model.learn(total_timesteps=300*10, log_interval=10)
    plt.close(fig=None)
    model.save("temp_model")
    # train in random env
    eps = 1600
    p_predator = 0.2
    episode_reward_history = []
    for ep in range(eps):
        random_number = random.random()
        has_predator = random_number < p_predator
        # has_predator = ep % 2 == 0
        env = Environment(e=2, has_predator=has_predator, max_step=300, env_type="train", predator_speed=0.2)
        model = SAC.load("temp_model", env=env)
        model.learn(total_timesteps=300, log_interval=1)
        episode_reward_history.append(env.episode_reward_history[0])
        model.save("temp_model")
        plt.close(fig=None)
        env.close()
    plot(episode_reward_history,name="final2")
    model.save("final2")

def SAC_train():
    env = Environment("17_09", freq=100, has_predator=False, max_step=300)
    env.reset()
    sigma = 0.1 # Adjust this value to increase the noise level
    # 0.24-0.23
    action_noise_mean_scale = 0.24 * 0.3
    action_noise = NormalActionNoise(mean=action_noise_mean_scale*np.ones(env.action_space.shape), sigma=sigma * np.ones(env.action_space.shape))
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
    plot(env.episode_reward_history, name="step_1")
    model.save("step_1")
    env.close()

def plot(data, name ="result", window = 100):
    import matplotlib.pyplot as plt
    import pandas as pd
    data_series = pd.Series(data)
    smooth_data = data_series.rolling(window=window).mean()
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
    env = Environment(e=5, freq=100, has_predator=True, max_step=2500, env_type="train"
                      , predator_speed=0.2)
    loaded_model = SAC.load("results_entropy/e5r10_2.zip", env=env)
    obs, _ = env.reset()
    #obs, _ = wrapped_env.reset()
    done = False
    tr = False
    all_action = []
    while not (done or tr):
        #if
        action, _states = loaded_model.predict(obs, deterministic=True)
        all_action.append(copy.deepcopy(action[0]))
        noise_level = 0
        rand_noise = np.random.randn(*action.shape) * noise_level
        action += rand_noise
        obs, reward, done, tr, _ = env.step(action)
        #env.show()
        if done:
            # print(F"Mean of actions is: {np.mean(all_action)}")
            # obs,_ = wrapped_env.reset()
            env.close()
    env.close()
    plt.close('all')
    plt.hist(all_action, label='Actions')
    plt.xlabel('action value')
    plt.ylabel('count')
    plt.legend()
    plt.title('Actions Hist')
    plt.show()
    plt.savefig("temp")

def result_visualization2():
    env = Environment(e=2, freq=100, has_predator=True, max_step=300, predator_speed=0.2, env_type="test")
    loaded_model = SAC.load("e2r10.zip", env)
    scores = []
    for i in range(100):
        obs,_ = env.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _states = loaded_model.predict(obs, deterministic=True)
            noise_level = 0.5
            rand_noise = np.random.randn(*action.shape) * noise_level
            action += rand_noise
            obs, reward, done, tr, _ = env.step(action)
            score += reward
            #env.show()
            # obs,_ = wrapped_env.reset()
        scores.append(score)
    plot(scores,"SACe=2_train", window=10)
    env.close()

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
    # result_visualization2()
    # random_policy()
    #evaluate()
    #check_prediction()
    #PPO_train()
    # random_env_training()
    # TD3_train_random_env()