"""Humanoid on-policy comparison using the completed HW2 PG implementation."""
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent / "增强学习_95065_2026Fall_HW2"))
import gym
import numpy as np
import torch
from gym.wrappers import ClipAction, RescaleAction, RecordEpisodeStatistics
from sjtuRL2026.agents.pg_agent import PGAgent
from sjtuRL2026.infrastructure import pytorch_util as ptu

OUT = Path(__file__).resolve().parent
SETTINGS = dict(env="Humanoid-v4", seed=1, total_steps=500000, batch_steps=5000,
                n_layers=3, layer_size=256, discount=0.99, learning_rate=3e-4,
                baseline_learning_rate=1e-3, baseline_gradient_steps=50,
                gae_lambda=0.95, eval_episodes=10)


def make_env():
    return RecordEpisodeStatistics(ClipAction(RescaleAction(gym.make("Humanoid-v4"), -1, 1)))


def trajectory(env, actor):
    obs = env.reset()
    observations, actions, rewards, terminals = [], [], [], []
    for _ in range(1000):
        action = actor.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        terminals.append(float(done))
        obs = next_obs
        if done:
            break
    terminals[-1] = 1.0
    return [np.asarray(x, dtype=np.float32) for x in (observations, actions, rewards, terminals)]


def main():
    torch.set_num_threads(1)
    np.random.seed(1)
    torch.manual_seed(1)
    ptu.init_gpu(False)
    env, eval_env = make_env(), make_env()
    env.reset(seed=1)
    eval_env.reset(seed=10001)
    agent = PGAgent(env.observation_space.shape[0], env.action_space.shape[0], False,
                    n_layers=3, layer_size=256, gamma=0.99, learning_rate=3e-4,
                    use_baseline=True, use_reward_to_go=True, normalize_advantages=True,
                    baseline_learning_rate=1e-3, baseline_gradient_steps=50, gae_lambda=0.95)
    (OUT / "pg_settings.json").write_text(json.dumps(SETTINGS, indent=2))
    started = time.time()
    with (OUT / "humanoid_pg.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "eval_return", "eval_std", "elapsed_seconds"])
        writer.writeheader()

        def evaluate(step):
            returns = [float(trajectory(eval_env, agent.actor)[2].sum()) for _ in range(10)]
            row = dict(step=step, eval_return=float(np.mean(returns)), eval_std=float(np.std(returns)),
                       elapsed_seconds=time.time() - started)
            writer.writerow(row)
            f.flush()
            print(json.dumps(row), flush=True)

        step, next_eval = 0, 10000
        evaluate(0)
        while step < 500000:
            paths, count = [], 0
            while count < min(5000, 500000 - step):
                path = trajectory(env, agent.actor)
                paths.append(path)
                count += len(path[2])
            agent.update(*[[p[i] for p in paths] for i in range(4)])
            step += count
            if step >= next_eval or step >= 500000:
                evaluate(step)
                next_eval = (step // 10000 + 1) * 10000
    env.close()
    eval_env.close()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
