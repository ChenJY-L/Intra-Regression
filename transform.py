import numpy as np
import pandas as pd
import torch


class IntraTransform:
    wave_length = ['1050', '1219', '1314',
                   '1409', '1550', '1609']

    def __init__(self, df: pd.DataFrame, prob=0.5):
        self.df = df
        self.prob = prob
        self.intra_list = np.sort(np.unique(df['intra'].to_list())).squeeze()

    def __call__(self, x, y):
        sigma = np.random.rand()

        if sigma < self.prob:
            current = self.locateData(x)
            intra = current['intra'].to_numpy()[0]
            HB = current['HB'].to_numpy()[0]
            cg = current['cg'].to_numpy()[0]
            intra2 = self.getNearestIntra(intra)
            x2 = self.getData(intra2[0], HB, cg)
            res = self.interp(intra, intra2, x, x2)
            return torch.tensor(res).float(), torch.tensor(y).float()

        else:
            return torch.tensor(x).float(), torch.tensor(y).float()

    def locateData(self, x):
        index = None
        # 查找数据的索引
        for i, wavelength in enumerate(self.wave_length):
            r = self.df[wavelength] == x[i]
            if index is None:
                index = r
            else:
                index = index & r

        # 查找对应的Intra数值
        return self.df.loc[index, :]

    def getData(self, intra, hb, cg):
        index = (self.df['intra'] == intra) & (self.df['HB'] == hb) & (self.df['cg'] == cg)
        value = self.df.loc[index].values
        return value[0, 3:]

    def getNearestIntra(self, intra):
        index = np.where(self.intra_list == intra)[0]

        if index == 0:
            return self.intra_list[1]

        elif index == (len(self.intra_list) - 1):
            return self.intra_list[len(self.intra_list) - 1]

        else:
            return self.intra_list[index + 1]

    def interp(self, x1, x2, y1, y2):
        """
        插值计算光谱
        :param x1: 自变量1
        :param x2: 自变量2
        :param y1: 因变量1
        :param y2: 因变量2
        :return: 插值结果光谱
        """
        y = [0] * 6
        xp = [x1, x2[0]]

        for i in range(len(self.wave_length)):
            yp = [y1[i], y2[i]]
            y[i] = np.interp(np.mean(xp), xp, yp)

        return y


if __name__ == '__main__':
    file_path = 'data/Intra_CLS1.xlsx'
    df1 = pd.read_excel(file_path)
    t = IntraTransform(df1, prob=1)
    testx = df1.iloc[150, 3:].to_list()
    testy = df1.iloc[150, 2]
    print(testx)
    print(t(testx, testy))