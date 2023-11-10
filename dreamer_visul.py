import numpy as np
import torch
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from prey_env import gym_Environment_D as Environment
from dreamerv2.utils.wrapper import CustomGymEnv, OneHotAction
from dreamerv2.training.config import CustomEnvConfig


def visulize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #env = Environment(e=2, has_predator="True", predator_speed=0.2, max_step=5000, env_type="train")
    env = Environment(e=2, has_predator=True, max_step=300, env_type="train", predator_speed=0.2)
    env = OneHotAction(CustomGymEnv(env))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    #saved_dict = torch.load("results/Predator_prey_bins_continus/models/models_best.pth")
    saved_dict = torch.load("models_best.pth", map_location=torch.device('cpu'))
    config = CustomEnvConfig(
        env="Predator_prey_bins",
        obs_shape=obs_shape,
        action_size=action_size
    )

    if config.rssm_type == 'continuous':
        stoch_size = config.rssm_info['stoch_size']
    elif config.rssm_type == 'discrete':
        category_size = config.rssm_info['category_size']
        class_size = config.rssm_info['class_size']
        stoch_size = category_size * class_size
    deter_size = config.rssm_info['deter_size']
    embedding_size = config.embedding_size
    rssm_node_size = config.rssm_node_size
    modelstate_size = stoch_size + deter_size

    ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(device).eval()
    ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(device).eval()
    ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor,
                                      config.expl).to(device).eval()
    rssm = RSSM(action_size, rssm_node_size, embedding_size, device, config.rssm_type, config.rssm_info).to(device).eval()

    rssm.load_state_dict(saved_dict["RSSM"])
    ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
    ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
    ActionModel.load_state_dict(saved_dict["ActionModel"])
    scores = []
    for i in range(100):  #epsoid number
        obs, score = env.reset(), 0
        done = False
        trun = False
        prev_rssmstate = rssm._init_rssm_state(1)
        prev_action = torch.zeros(1, action_size).to(device)
        while not (done or trun):
            with torch.no_grad():
                embed = ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
                _, posterior_rssm_state = rssm.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                model_state = rssm.get_model_state(posterior_rssm_state)
                action, _ = ActionModel(model_state)
                prev_rssmstate = posterior_rssm_state
                prev_action = action
            next_obs, rew, done, trun, _ = env.step(action.squeeze(0).cpu().numpy())
            # env.render()
            score += rew
            obs = next_obs
        scores.append(score)
    plot(scores, "dreamer_e2_train")


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

if __name__ =="__main__":
    visulize()