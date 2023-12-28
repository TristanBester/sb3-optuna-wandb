import gymnasium as gym
import optuna
from stable_baselines3.common.callbacks import EvalCallback
from wandb.sdk.wandb_run import Run


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
