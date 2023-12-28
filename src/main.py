import os
import time
from multiprocessing import Pool

import optuna
from dotenv import load_dotenv
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from exp import blackbox
from report import create_parallel_coords_plot


def work(pid: int):
    if pid != 0:
        # Wait for the first process to initialise the database
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
    create_parallel_coords_plot(
        exp_name="optuna_exp_3",
    )
