# Coding: UTF-8

import numpy as np

import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget

from config import RegressionConfig, ClsConfig
from models_reg import ResNet1D
from models_cls import CLSRes

def reshape_transform(tensor):
    image_with_single_row = tensor[:, None, :, :]
    # Lets make the time series into an image with 16 rows for easier visualization on screen later
    target_size = 4, tensor.size(1)
    return torch.nn.functional.interpolate(image_with_single_row, target_size, mode='bilinear')


def getClsCAM(model, input_tensor, category):

    cam = GradCAM(model, [model.backbone_net2.conv2], reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(category)]
    map = cam(input_tensor.unsqueeze(0), targets)
    return map

def test_CLS():
    c = ClsConfig()
    model = CLSRes(c.num_classes, c.feature_size, c.hidden_size)
    model.load_state_dict(torch.load(c.save_path))
    model.eval()
    input_data = np.array([[-0.768718367, 0.077303164, 0.191715744,
                            4.456062486, 3.557174579, 2.449878984],
                           [-1.051016012, -0.253492721, -0.065104923,
                            3.382362281, 2.468472749, 1.387292059],
                           [-1.125627081, -0.294017213, -0.145552968,
                            2.355356848, 1.557701141, 0.520258978]])

    for i in range(c.num_classes):
        Map = getClsCAM(model, torch.tensor(input_data[i]).float(), i)
        print(Map)


if __name__ == "__main__":
    test_CLS()
