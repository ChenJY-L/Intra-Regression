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
- [ ] 寻找新的方法缩减不必要的模型分支
- [x] 完善Bayessian优化程序
- [x] 测试数据增强效果
- [ ] 考虑血糖数据集的数据增强问题
- [ ] 测试集更换完全没看过的intra样本
- [x] 验证真intra样本的影响
- [ ] 计算intra之间的插值权重 or 拟合一个吸光度-散射变化曲线

## 问题 📌
- Intra浓度表示不同的实验条件
- 糖浓度变化引起$u_a$和$u_s$变化
- HB浓度变化也会引起$u_a$和$u_s$变化
