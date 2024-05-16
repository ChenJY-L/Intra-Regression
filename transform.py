import numpy as np
import pandas as pd
import torch


class IntraTransform(torch.nn.Module):
    wave_length = ['1050', '1219', '1314',
                   '1409', '1550', '1609']

    def __init__(self, df: pd.DataFrame, prob=0.5):
        super(IntraTransform, self).__init__()
        self.df = df
        self.prob = prob
        self.intra_list = np.sort(np.unique(df['intra'].to_list())).squeeze()

    def forward(self, x, y):
        sigma = np.random.rand()

        if sigma < self.prob:
            current = self.locateData(x)
            intra = current['intra'].to_numpy()[0]
            HB = current['HB'].to_numpy()[0]
            cg = current['cg'].to_numpy()[0]
            intra2 = self.getNearestIntra(intra)
            x2 = self.getData(intra2, HB, cg)

            if x2 is not None and intra != intra2:
                res = self.interp(intra, intra2, x, x2)
                x = res

        return self.toTensor(x, y)

    def toTensor(self, x, y):
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

        if len(value) == 0:
            # 查找为空
            return None

        return value[0, 3:]

    def getNearestIntra(self, intra):
        index = np.searchsorted(self.intra_list, intra)
        if index == 0:
            return self.intra_list[index]
        # elif index == len(self.intra_list):
        #     return self.intra_list[-1]
        else:
            return self.intra_list[index - 1]

    def interp(self, x1, x2, y1, y2):
        y = [0] * 6
        xp = [x2, x1]
        xf = np.mean(xp)
        for i in range(len(self.wave_length)):
            yp = [y2[i], y1[i]]
            y[i] = np.interp(xf, xp, yp)

        return y


if __name__ == '__main__':
    file_path = 'data/Intra_CLS1.xlsx'
    output_path = 'data/Aug_Intra.xlsx'
    df1 = pd.read_excel(file_path)
    df2 = df1.copy(deep=True)

    t = IntraTransform(df1, prob=1)
    for i, row in df1.iterrows():
        x = row[3:].to_numpy()
        y = row.iloc[2]
        intra = np.mean([row.iloc[0], t.getNearestIntra(row.iloc[0])])

        aug_x, aug_y = t(x, y)
        new_row = np.concatenate(([intra, row.iloc[1], y], aug_x.numpy().reshape(-1)))
        if intra == row.iloc[0]:
            continue
        df2.loc[len(df2)] = new_row


    df2.to_excel(output_path, index=False)
