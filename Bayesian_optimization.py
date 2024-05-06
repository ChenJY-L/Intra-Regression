import numpy as np
from train_reg import train, RegressionConfig

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
# from ax.utils.tutorials.cnn_utils import train, evaluate


def train_evaluate(parameters):
    config = RegressionConfig()
    config.batch_size = parameters.get("batch_size", 16)
    # config.learning_rate = parameters.get("learning_rate", 1e-6)
    # config.momentum = parameters.get("momentum", 1e-2)
    config.hidden_size = parameters.get("hidden_size", 66) * 2

    RMSE = train(config, False)
    return RMSE


if __name__ == "__main__":
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "hidden_size", "type": "range", "bounds": [64, 75]},
            {"name": "batch_size", "type": "range", "bounds": [8, 32]},
        ],

        evaluation_function=train_evaluate,
        objective_name='accuracy',
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)

    best_objectives = np.array([[trial.objective_mean * 100 for trial in experiment.trials.values()]])

    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )
    render(best_objective_plot)

    render(plot_contour(model=model, param_x='batch_size', param_y='hidden_size', metric_name='accuracy'))
