import os
import time
from multiprocessing import Pool

import optuna
from optuna.samplers import TPESampler
from wandb.apis import reports as wr

import wandb


def objective_fn(trial: optuna.Trial):
    """f(n) = m_x + c"""
    m = trial.suggest_float("m", -10, 10)
    c = trial.suggest_int("c", -10, 10)

    print("In experiment...")

    config = {"m": m, "c": c}
    run = wandb.init(
        project="optuna_exp_3",
        name=f"trial_{trial._trial_id}",
        config=config,
        reinit=True,
    )
    print("Experiment running...")

    # simulate trainining
    for step in range(60):
        y_hat = m * step + c
        run.log(
            {
                "step": step,
                "y_hat": y_hat,
            }
        )
        time.sleep(1)
    return 100 * m + c


def work(id_):
    time.sleep(1 + id_ * 2)
    print(f"Starting work... {id_}")
    sampler = TPESampler(seed=id_)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="test",
        load_if_exists=True,
        storage="sqlite:///test.db",
    )
    study.optimize(
        func=objective_fn,
        n_trials=3,
    )


if __name__ == "__main__":
    # Disable wandb logging to stdout
    os.environ["WANDB_SILENT"] = "true"

    # Run experiments in parallel
    with Pool(processes=5) as pool:
        pool.map(work, range(5))

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
                            wr.PCColumn("c::m"),
                            wr.PCColumn("c::c"),
                            wr.PCColumn("y_hat"),
                        ],
                        layout={"w": 24, "h": 9},
                    ),
                ]
            )
        ],
    )
    report.save()
