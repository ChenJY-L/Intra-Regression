# Coding: UTF-8

import numpy as np

import torch
import torch.nn as nn

from torchcam.methods import GradCAM, SmoothGradCAMpp

from config import RegressionConfig
from regression_models import ResNet1D


def getCAM(model, input_tensor):
    with SmoothGradCAMpp(model, target_layer=model.fc2, num_samples=1, input_shape=(6,)) as cam_extractor:

        out = model(input_tensor.unsqueeze(0))

        activation_map = cam_extractor(1, out)
    return activation_map


if __name__ == "__main__":
    c = RegressionConfig()
    model = ResNet1D(c.feature_size, c.hidden_size, c.output_size, c.num_layers, c.dropout_prob)
    model.load_state_dict(torch.load(c.save_path))
    model.eval()
    input_data = np.array([-0.768718367, 0.077303164, 0.191715744,
                            4.456062486, 3.557174579, 2.449878984])

    Map = getCAM(model, torch.tensor(input_data).float())
    print(Map)
