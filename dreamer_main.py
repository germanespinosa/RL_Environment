import argparse
import os
import torch
import numpy as np
from dreamerv2.utils.wrapper import CustomGymEnv, OneHotAction
from dreamerv2.training.config import CustomEnvConfig
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator
from gym_bins_env import Environment
from tqdm import tqdm


def main(args):
    env = Environment(e=2, has_predator="True", predator_speed=0.2, max_step=300)
    env_name = "Predator_prey_bins"
    exp_id = args.id
    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')  # dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    print('using :', device)

    env = OneHotAction(CustomGymEnv(env))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]

    batch_size = args.batch_size
    seq_len = args.seq_len

    config = CustomEnvConfig(
        env="Predator_prey_bins",
        obs_shape=obs_shape,
        action_size=action_size,
        seq_len=seq_len,
        batch_size=batch_size,
        model_dir=model_dir,
    )

    trainer = Trainer(config, device)
    """training loop"""
    print('...training...')
    train_metrics = {}
    trainer.collect_seed_episodes(env)
    obs, score = env.reset(), 0
    done = False
    prev_rssmstate = trainer.RSSM._init_rssm_state(1)
    prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
    episode_actor_ent = []
    scores = []
    rewards = []
    best_mean_score = -400
    best_save_path = os.path.join(model_dir, 'models_best.pth')

    for iter in tqdm(range(1, trainer.config.train_steps), desc="Training Progress"):
        if iter % trainer.config.train_every == 0:
            train_metrics = trainer.train_batch(train_metrics)
        if iter % trainer.config.slow_target_update == 0:
            trainer.update_target()
        if iter % trainer.config.save_every == 0:
            trainer.save_model(iter)
        with torch.no_grad():
            embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device))
            _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
            model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
            action, action_dist = trainer.ActionModel(model_state)
            action = trainer.ActionModel.add_exploration(action, iter).detach()
            action_ent = torch.mean(action_dist.entropy()).item()
            episode_actor_ent.append(action_ent)

        next_obs, rew, done, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        score += rew

        if done or truncated:
            trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done, truncated)
            train_metrics['train_rewards'] = score
            train_metrics['action_ent'] = np.mean(episode_actor_ent)
            scores.append(score)
            rewards.append(score)
            if len(scores) > 50:
                scores.pop(0)
                current_average = np.mean(scores)
                if current_average > best_mean_score:
                    best_mean_score = current_average
                    print('saving best model with mean score : ', best_mean_score)
                    save_dict = trainer.get_save_dict()
                    torch.save(save_dict, best_save_path)
            obs, score = env.reset(), 0
            done = False
            prev_rssmstate = trainer.RSSM._init_rssm_state(1)
            prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
            episode_actor_ent = []
        else:
            trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done, truncated)
            obs = next_obs
            prev_rssmstate = posterior_rssm_state
            prev_action = action
    plot(rewards, name="Dreamer")

def plot(data, name ="result"):
    import matplotlib.pyplot as plt
    import pandas as pd
    data_series = pd.Series(data)
    smooth_data = data_series.rolling(window=50).mean()
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



if __name__ == "__main__":
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default='d_10_100', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    main(args)