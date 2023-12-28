import gymnasium as gym
import optuna
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn import DQN

from callback import TrialEvalCallback


def blackbox(trial: optuna.Trial):
    """Blackbox objective function for optimization."""
    # Don't monitor the training environment
    config = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
        "target_update_interval": trial.suggest_int(
            "target_update_interval", 100, 10_000
        ),
    }
    run = wandb.init(
        project="optuna_exp_3",
        name=f"trial_{trial._trial_id}",
        config=config,
        reinit=True,
    )

    env = gym.make("CartPole-v1")
    model = DQN(env=env, policy="MlpPolicy", **config)

    # Monitor the evaluation environment
    eval_env = Monitor(gym.make("CartPole-v1"))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        run,
        n_eval_episodes=25,
        eval_freq=10_000,
        deterministic=True,
    )
    model.learn(
        total_timesteps=500_000,
        callback=eval_callback,
    )
    wandb.finish()
    return eval_callback.last_mean_reward
