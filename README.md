# Intra-Regression
## Introduction 📖
A regression model using six wavelengths of absorbance, for predicting the glucose in intra.
The glucose simulated in MonteCarlo method using **MCX** and **CUDAMCML**. 
We randomly set HB, intra and glucose in MC simulation. 

## Function ⌨️
- main.py: Train a regression model.
- models.py: Classes for regression model.

## Idea 🔥
1. Build a model for classification of Intra 
2. Based on classification, evaluate the performance of PLS and NN for the same Intra

## TODO ✅
- [ ] Write code for PLS
- [ ] Write a class for analyze dataset

## 问题 📌
- Intra浓度表示不同的实验条件
- 糖浓度变化引起$u_a$和$u_s$变化
- HB浓度变化也会引起$u_a$和$u_s$变化

## 分析
- 决策系数R2 接近 1, 

## Train log
date:5/6

| 更新         | RMSE | 
|------------|------|
| epoch=3000 | a    |
