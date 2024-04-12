import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from models import *
from evaluate import *
from train import IntraDataset, Config


if __name__ == "__main__":
    c = Config()
    model = ResNet1D(c.feature_size, c.hidden_size, c.output_size, c.num_layers, c.dropout_prob)
    model.load_state_dict(torch.load(c.save_path))

    model.eval()

    test_input = torch.randn(1, 6)
    torch.onnx.export(model, test_input, c.model_name + ".onnx")
