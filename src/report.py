from wandb.apis import reports as wr


def create_parallel_coords_plot(exp_name: str):
    # Create a parallel coordinates plot
    report = wr.Report(
        exp_name,
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
