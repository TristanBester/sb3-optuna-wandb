import os
import time
from multiprocessing import Pool

import gymnasium as gym
import optuna
from dotenv import load_dotenv
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn import DQN
from wandb.apis import reports as wr
from wandb.sdk.wandb_run import Run

import wandb


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        run: Run,
        n_eval_episodes: int = 5,
        eval_freq: int = 1000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.run = run
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                self.run.log(
                    {
                        "samples": self.n_calls,
                        "reward": self.last_mean_reward,
                        "pruned": 1,
                    }
                )
                return False
            self.run.log(
                {
                    "samples": self.n_calls,
                    "reward": self.last_mean_reward,
                    "pruned": 0,
                }
            )
        return True


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
        total_timesteps=200_000,
        callback=eval_callback,
    )
    wandb.finish()
    return eval_callback.last_mean_reward


def work(pid: int):
    if pid != 0:
        time.sleep(30)
    print(f"Starting work... {pid}")
    sampler = TPESampler(seed=pid)
    pruner = MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=100_000,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="test",
        load_if_exists=True,
        storage=os.environ.get("DB_URL"),
    )
    study.optimize(
        func=blackbox,
        n_trials=10,
    )


if __name__ == "__main__":
    load_dotenv()
    # Disable wandb logging to stdout
    os.environ["WANDB_SILENT"] = "true"

    # Run experiments in parallel
    with Pool(processes=6) as pool:
        pool.map(work, range(6))

    # Create a parallel coordinates plot
    report = wr.Report(
        "optuna_exp_3",
        title="Hyperparameter Optimization",
        description="Parellel Coordinates Plot",
        blocks=[
            wr.PanelGrid(
                panels=[
                    wr.ParallelCoordinatesPlot(
                        columns=[
                            # c:: prefix accesses config variable
                            wr.PCColumn("c::learning_rate"),
                            wr.PCColumn("c::target_update_interval"),
                            wr.PCColumn("reward"),
                        ],
                        layout={"w": 24, "h": 9},
                    ),
                ]
            )
        ],
    )
    report.save()
