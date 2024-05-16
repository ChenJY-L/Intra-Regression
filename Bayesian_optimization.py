import numpy as np
from train_reg import train, RegressionConfig

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
# from ax.utils.tutorials.cnn_utils import train, evaluate


def train_evaluate(parameters):
    config = RegressionConfig()
    config.num_epochs = 1000
    config.batch_size = parameters.get("batch_size", 16)
    config.learning_rate = parameters.get("learning_rate", 1) * 1e-6
    # config.momentum = parameters.get("momentum", 1e-2)
    config.hidden_size = parameters.get("hidden_size", 66) * 2
    config.num_layers = parameters.get("num_layers", 3)

    RMSE = train(config, False)
    return RMSE


if __name__ == "__main__":
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "hidden_size",
                "type": "range",
                "bounds": [50, 82],
                'value_type': 'int'
            },
            {"name": "num_layers",
             "type": "range",
             "bounds": [1, 5],
             'value_type': 'int'
             },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [8, 32],
                'value_type': 'int'
            },
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1, 10],
                'value_type': 'float'
            },
        ],

        evaluation_function=train_evaluate,
        objective_name='RMSE',
        minimize=True,
        total_trials=10,
    )

    print("Best")
    print(best_parameters)
    means, covariances = values
    print(means)
    # print(covariances)

    best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])

    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="RMSE, %",
    )
    render(best_objective_plot)

    render(plot_contour(model=model, param_x='num_layers', param_y='hidden_size', metric_name='RMSE'))

