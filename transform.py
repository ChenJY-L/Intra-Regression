import numpy as np
import pandas as pd
import torch


class LinearTransform(torch.nn.Module):
    wave_length = ['1050', '1219', '1314',
                   '1409', '1550', '1609']

    def __init__(self, df: pd.DataFrame, prob=0.5):
        super(LinearTransform, self).__init__()
        self.df = df
        self.prob = prob
        self.intra_list = np.sort(np.unique(df['intra'].to_list())).squeeze()

    def forward(self, x, y):
        sigma = np.random.rand()
        intra_res = None

        if sigma < self.prob:
            current = self.locateData(x)
            intra = current['intra'].to_numpy()[0]
            HB = current['HB'].to_numpy()[0]
            cg = current['cg'].to_numpy()[0]
            intra2 = self.getNearestIntra(intra)
            x2 = self.getData(intra2, HB, cg)

            if x2 is not None and intra != intra2:
                res, intra_res = self.interp(intra, intra2, x, x2)
                x = res

        return x, y, intra_res
        # return self.toTensor(x, y)

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

    def getNearestIntra(self, intra):

        index = np.searchsorted(self.intra_list, intra)
        if index == 0:
            return self.intra_list[index]
        # elif index == len(self.intra_list):
        #     return self.intra_list[-1]
        else:
            return self.intra_list[index - 1]

    def getData(self, intra, hb, cg):
        index = (self.df['intra'] == intra) & (self.df['HB'] == hb) & (self.df['cg'] == cg)
        value = self.df.loc[index].values

        if len(value) == 0:
            # 查找为空
            return None

        return value[0, 3:]

    def interp(self, x1, x2, y1, y2):
        y = [0] * 6
        xp = [x2, x1]
        xf = np.mean(xp)
        for i in range(len(self.wave_length)):
            yp = [y2[i], y1[i]]
            y[i] = np.interp(xf, xp, yp)

        return np.array(y), xf

    def add_noise(self, spec, num):
        res = np.ones((num, 1)) * spec
        noise = np.random.normal(loc=1, scale=0.1, size=(num, 1))
        res = res * noise
        return res

    def export(self, output_path='data/Aug_Intra2.xlsx'):
        df = self.df.copy()

        # Interp Aug
        for i, row in self.df.iterrows():
            spec = row[3:].to_numpy()
            cg = row.iloc[2]
            HB = row.iloc[1]
            # HB = np.ones((self.ratio, 1)) * HB
            aug_x, aug_y, intra_res = self.forward(spec, cg)
            if intra_res is None:
                continue
            new_rows = np.concatenate([np.array([intra_res, HB, aug_y]), aug_x])
            df.loc[len(df)] = new_rows

        # Noise Aug
        ratio = 2
        for i, row in df.iterrows():
            spec = row[3:].to_numpy()
            aug_spec = self.add_noise(spec, ratio)
            new_rows = np.concatenate(([np.tile(row[0:3].to_numpy(), (ratio, 1)), aug_spec]), axis=1)
            new_rows = pd.DataFrame(new_rows, columns=df.columns)
            df = pd.concat([df, new_rows], ignore_index=True)

        df.to_excel(output_path, index=False)


class RandomLinearTransform(LinearTransform):

    def __init__(self, df, ratio=10, prob=0.5):
        super(RandomLinearTransform, self).__init__(df, prob)
        self.ratio = ratio

    def forward(self, x, y):
        sigma = np.random.rand()

        intra_res = None
        if sigma < self.prob:
            current = self.locateData(x)
            intra = current['intra'].to_numpy()[0]
            HB = current['HB'].to_numpy()[0]
            cg = current['cg'].to_numpy()[0]
            intra2 = self.getNearestIntra(intra)
            x2 = self.getData(intra2, HB, cg)

            if x2 is not None and intra != intra2:
                res, intra_res = self.interp(intra, intra2, x, x2)
                x = res
                y = np.ones((self.ratio, 1)) * y

        return x, y, intra_res

    def interp(self, x1, x2, y1, y2):
        y = np.zeros((len(self.wave_length), self.ratio))
        xp = [x2, x1]
        # xf = np.abs(np.random.normal(loc=0, scale=(x1 - x2)/2, size=(1, self.ratio))) + x2
        xf = np.ones((1, self.ratio)) * np.mean(xp)

        noise = np.random.normal(loc=0, scale=0.1, size=(1, self.ratio))
        for i in range(len(self.wave_length)):
            yp = [y2[i], y1[i]]
            y[i, :] = np.interp(xf, xp, yp) * (1 + noise)

        return y, xf

    def export(self, output_path='data/Aug_Intra.xlsx'):
        df = self.df.copy(deep=True)

        for i, row in self.df.iterrows():
            spec = row[3:].to_numpy()
            cg = row.iloc[2]
            HB = row.iloc[1]
            HB = np.ones((self.ratio, 1)) * HB
            aug_x, aug_y, intra_res = self.forward(spec, cg)
            if intra_res is None:
                continue
            new_rows = np.concatenate(([intra_res.T, HB, aug_y, aug_x.T]), axis=1)
            new_rows = pd.DataFrame(new_rows, columns=df.columns)
            df = pd.concat([df, new_rows], ignore_index=True)

        df.to_excel(output_path, index=False)


if __name__ == '__main__':
    file_path = 'data/Intra_CLS1.xlsx'
    output_path = 'data/Aug_Intra2.xlsx'
    # df1 = pd.read_excel(file_path)
    # df2 = df1.copy(deep=True)
    #
    # t = LinearTransform(df1, prob=1)
    # for i, row in df1.iterrows():
    #     x = row[3:].to_numpy()
    #     y = row.iloc[2]
    #     intra = np.mean([row.iloc[0], t.getNearestIntra(row.iloc[0])])
    #
    #     aug_x, aug_y = t(x, y)
    #     new_row = np.concatenate(([intra, row.iloc[1], y], aug_x.numpy().reshape(-1)))
    #     if intra == row.iloc[0]:
    #         continue
    #     df2.loc[len(df2)] = new_row
    #
    # df2.to_excel(output_path, index=False)

    df = pd.read_excel(file_path)
    t = LinearTransform(df, prob=1)
    t.export(output_path)
