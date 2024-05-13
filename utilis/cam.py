# Coding: UTF-8

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget, RawScoresOutputTarget

from config import RegressionConfig, ClsConfig
from models_reg import ResNet1D, RegNet
from models_cls import CLSRes
from datasets import *

def reshape_transform(tensor):
    image_with_single_row = tensor[:, None, :, :]
    # Lets make the time series into an image with 16 rows for easier visualization on screen later
    target_size = 6, tensor.size(1)
    return torch.nn.functional.interpolate(image_with_single_row, target_size, mode='bilinear')


def getClsCAM(model, input_tensor, category):
    cam = GradCAM(model, [model.backbone_net2.conv2], reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(category)]
    map = cam(input_tensor.unsqueeze(0), targets)
    return map


def getResCAM(model, input_tensor):
    # print(model.backbone_net[2].conv2)
    cam = GradCAM(model, [model.backbone_net[2].conv2], reshape_transform=reshape_transform)
    targets = [RawScoresOutputTarget()]
    map = cam(input_tensor.unsqueeze(0), targets)
    return map


def test_cls():
    c = ClsConfig()
    model = CLSRes(c.num_classes, c.feature_size, c.hidden_size)
    model.load_state_dict(torch.load(c.save_path))
    model.eval()
    input_data = np.array(
        # Intra 2   HB 0    CG 0
        [[-0.768718367, 0.077303164, 0.191715744,
          4.456062486, 3.557174579, 2.449878984],
         # Intra 5  HB 20   CG 0
         [-1.051016012, -0.253492721, -0.065104923,
          3.382362281, 2.468472749, 1.387292059],
         # Intra 10 HB 0    CG 0
         [-1.125627081, -0.294017213, -0.145552968,
          2.355356848, 1.557701141, 0.520258978]])

    print("Intra data ")
    for i in range(c.num_classes):
        print(f"Intra: {c.idx_to_class[i]}%")
        Map = getClsCAM(model, torch.tensor(input_data[i]).float(), i)
        print(Map)


def test_reg():
    c = RegressionConfig()
    model = RegNet(c.feature_size, c.hidden_size, c.output_size)
    model.eval()

    dataset = read_data(c.data_path)
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    data_export = []
    for i in range(len(dataset)):
        input_data = torch.tensor(dataset[i, 3:]).float()
        predict = model(input_data)
        map = getResCAM(model, input_data)
        row_data = np.hstack((dataset[i], map.squeeze(), predict.numpy()))
        data_export.append(row_data)

    df = pd.DataFrame(data_export)
    df.columns = ["Intra", "HB", "cg",
                  "x1050", "x1219", "x1314",
                  "x1409", "x1550", "x1609",
                  "cam1050", "cam1219", "cam1314",
                  "cam1409", "cam1550", "cam1609",
                  "cg_predict"]
    df.to_csv("reg.csv")


if __name__ == "__main__":
    # test_cls()
    test_reg()
