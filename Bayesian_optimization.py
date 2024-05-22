import numpy as np
from train_reg import train, RegressionConfig

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
# from ax.utils.tutorials.cnn_utils import train, evaluate


def get_config(parameters):
    config = RegressionConfig()
    config.num_epochs = 300
    config.batch_size = parameters.get("batch_size", 16)
    config.learning_rate = parameters.get("learning_rate", 1e-5)
    # config.momentum = parameters.get("momentum", 1e-2)
    config.hidden_size = parameters.get("hidden_size", 66) * 2
    config.num_layers = parameters.get("num_layers", 3)

    return config


def train_evaluate(parameters):
    config = get_config(parameters)
    RMSE, _ = train(config, show_status=False, test_extra=False)
    return RMSE


def test_model(parameters):
    config = get_config(parameters)
    _, model = train(config, show_status=False, test_extra=True)


# if __name__ == "__main__":
def main():
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "hidden_size",
                "type": "range",
                "bounds": [50, 100],
                'value_type': 'int'
            },
            {
                "name": "num_layers",
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
                "bounds": [1e-6, 1e-4],
                'value_type': 'float',
                "log_scale": True,
            },
        ],

        evaluation_function=train_evaluate,
        objective_name='RMSE',
        minimize=True,
        total_trials=40,
    )

    # /--------------------------------
    means, covariances = values
    print("Best")
    print(best_parameters)
    print(means)
    print(covariances)

    # /--------------------------------
    print("Train best parameters")
    test_model(best_parameters)

    # /--------------------------------
    best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(best_objectives, axis=1),
        # y=best_objectives,
        title="Model performance vs. # of iterations",
        ylabel="RMSE, %",
    )
    render(best_objective_plot)
    render(plot_contour(model=model, param_x='num_layers', param_y='hidden_size', metric_name='RMSE'))


if __name__ == "__main__":
    main()
