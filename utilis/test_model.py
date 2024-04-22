import pandas as pd

from regression_models import *
from utilis.evaluate import *
from train_regression import IntraDataset, RegressionConfig


if __name__ == "__main__":
    c = RegressionConfig()
    model = ResNet1D(c.feature_size, c.hidden_size, c.output_size, c.num_layers, c.dropout_prob)
    model.load_state_dict(torch.load(c.save_path))

    test_df = pd.read_excel(c.test_data_path)
    y_scaler = MinMaxScaler()
    test_df.iloc[:, 0] = y_scaler.fit_transform(test_df.iloc[:, 0].values.reshape(-1, 1))

    x_scaler = MinMaxScaler()
    test_df.iloc[:, 1:] = x_scaler.fit_transform(test_df.iloc[:, 1:])
    test_data = IntraDataset(test_df)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    all_test_predictions, all_test_targets = evaluate_model(model, test_loader, y_scaler)
    plot_predictions(all_test_predictions, all_test_targets)