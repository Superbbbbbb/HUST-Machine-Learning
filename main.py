# coding=utf-8
import numpy as np
import pandas as pd


class Adaboost:
    def __init__(self, base):
        self.base = base

    def normalization(self, data):
        for i in range(self.m):
            for j in range(self.n):
                data[i][j] = (data[i][j] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
        return data

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def logistic_regression(self, data, label, D, alpha=5.8, epochs=15000, epsilon=1e-5):
        theta = {'w': np.zeros((self.n, 1))}  # 初始化梯度
        h = self.sigmoid(np.dot(data, theta['w']))  # 初始化h
        for i in range(epochs):
            loss = (h - label) * D  # 损失函数
            theta['w'] -= alpha * np.dot(data.T, loss)  # 梯度
            h = self.sigmoid(np.dot(data, theta['w']))  # 更新h
            if alpha * np.sum(abs(loss)) < epsilon:  # 损失函数的模小于容错值
                break
        return theta

    def decision_stump(self, data, label, D):  # 生成决策树
        stump = {}  # 决策树桩
        minErr = 1  # 总最小错误率
        x = data.T
        for i in range(self.n):
            c = list(set(x[i]))  # 去掉重复特征值
            c.sort(reverse=False)  # 升序
            m = np.shape(c)[0]
            cnt = 0
            min_err = 1  # 当前feature的最小错误率
            for j in range(m - 1):
                split = (c[j] + c[j + 1]) / 2  # 分界线
                err, symbol = self.cal_err1(data, label, split, i, D)
                if err < min_err:  # 误差最小的分类器
                    cnt = 0
                    min_err = err
                    if err < minErr:
                        minErr = err
                        stump['feature'] = i  # 属性
                        stump['split'] = split  # 分界值
                        stump['symbol'] = symbol  # 大于还是小于
                        stump['err'] = err  # 错误率
                else:
                    cnt += 1
                if cnt == 5:
                    break
        return stump

    def cal_err0(self, r, label, d):
        err = 0
        for i in range(self.x):
            if r[i][0] != label[i][0]:
                err += d[i][0]  # 样本的权重作为错误率
        return err

    def cal_err1(self, data, label, split, j, d):  # 计算分类器的总误差
        err = 0
        for i in range(self.x):
            if (data[i][j] >= split and label[i][0] == 0.0) or (data[i][j] < split and label[i][0] == 1.0):
                err += d[i][0]  # 样本的权重作为错误率
        if err < 0.5:
            return err, -1  # 大于split
        else:
            return 1 - err, 1  # 小于split

    def adaboost(self, T, data, label):
        Dt = np.ones((self.x, 1)) / self.x  # 初始化权重
        h = []  # 基分类器
        for t in range(T):
            r = []  # 预测结果
            if self.base == 1:
                classifier = self.decision_stump(data, label, Dt)  # 生成决策树
                for j in range(self.x):
                    r.append([(self.pre(classifier, data[j]) + 1) / 2])
                if classifier['err'] > 0.45:
                    break
            else:
                classifier = self.logistic_regression(data, label, Dt)
                r = self.sigmoid(np.dot(data, classifier['w'])) >= 0.5
                classifier['err'] = self.cal_err0(r, label, Dt)
                if classifier['err'] >= 0.5:
                    break
            alpha = np.log((1 - classifier['err']) / classifier['err']) / 2  # 计算权重系数
            for j in range(self.x):
                Dt[j][0] *= np.exp(-1 * alpha * ((label[j][0] == r[j][0]) * 2 - 1))  # 根据预测结果更新权重
            for j in range(self.x):
                Dt[j][0] /= np.sum(Dt)  # 归一化权重
            classifier['alpha'] = alpha
            h.append(classifier)
        return h

    def fit(self, x_file, y_file):
        data = np.array(pd.read_csv(x_file, header=None).values.tolist())
        label = np.array(pd.read_csv(y_file, header=None).values.tolist())

        self.m, self.n = np.shape(data)
        self.y = int(self.m / 10)
        self.x = self.y * 9
        self.T = [1, 5, 10, 100]
        if self.base == 0:
            data = self.normalization(data)

        a = 0
        for t in self.T:
            for i in range(10):
                # 十折交叉验证手动实现
                self.test = np.array(range(self.y * i, self.y * (i + 1)))
                self.train = list(range(self.m))
                del self.train[self.m - self.y * (10 - i): self.m - self.y * (9 - i)]
                self.train = np.array(self.train)

                # Adaboost
                h = self.adaboost(t, data[self.train], label[self.train])
                r = self.fold_predict(h, data[self.test])

                temp = self.fold_accuracy(r, label[self.test])
                if temp > a:
                    a = temp
                    self.classifier = h
                df = pd.DataFrame(r)
                df.to_csv('./experiments/base%d_fold%d.csv' % (t, i + 1), header=False, index=False)
        return self

    def predict(self, x_file):
        data = pd.read_csv(x_file, header=None).values.tolist()
        data = np.array(data)

        if self.base == 0:
            data = self.normalization(data)
        result = []
        for i in range(self.m):
            r = 0
            for j in self.classifier:
                r += j['alpha'] * self.pre(j, data[i])  # 权重系数与分类结果的乘积
            result.append(r > 0)
        return result

    def accuracy(self, result, y_file):
        label = pd.read_csv(y_file, header=None).values.tolist()
        label = np.array(label)
        cnt = 0
        for i in range(self.m):
            if result[i] == label[i][0]:
                cnt += 1.0
        return cnt / self.m

    def fold_predict(self, h, data):  # 十折预测结果
        r = []
        for i in range(self.y):
            tmp = [self.test[i] + 1]
            p = 0
            for j in h:
                p += j['alpha'] * self.pre(j, data[i])
            tmp.append(int(p >= 0))
            r.append(tmp)
        return r

    def fold_accuracy(self, r, label):  # 十折预测精度
        cnt = 0
        for i in range(self.y):
            if r[i][-1] == label[i][0]:
                cnt += 1.0
        return cnt / self.y

    def pre(self, h, data):
        if self.base == 1:
            return h['symbol'] * ((h['split'] >= data[h['feature']]) * 2 - 1)
        else:
            return (self.sigmoid(np.dot(data, h['w'])) >= 0.5) * 2 - 1


if __name__ == '__main__':
    _base = input()
    _adaboost = Adaboost(_base)
    _adaboost.fit('./data.csv', './targets.csv')
    _result = _adaboost.predict('./data.csv')
    print(_adaboost.accuracy(_result, './targets.csv'))
