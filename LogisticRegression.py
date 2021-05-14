import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class LogisticRegression(object):

    def __init__(self, n_iter=5000, eta=0.0005, diff=None):
        # 梯度下降循环次数
        self.n_iter = n_iter
        # 学习率
        self.eta = eta
        # 提前停止的阈值
        self.diff = diff
        # 参数w，训练最终需要的结果
        self.w = None

    def _Fit(self, X_train, Y_train):
        """
        首先进行数据预处理.
        X_train 输入的是每个样本水平放置的训练集，在训练集最左侧需要添加一列1.
        (原因为了将偏置也放进参数w中，不懂的话需要自己取补补知识了)
        """
        X_train = self._X_preprocess(X_train)  # 方法①
        # 根据X的特征数初始化参数w
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05  # 随机初始化 默认shape为(n, 1)
        # 进行利用梯度下降的模型学习
        self._gradient_descent(self.w, Y_train, X_train)  # 方法②

    # 部分方法补充
    # 方法①
    def _X_preprocess(self, Xtrain):

        # 数据预处理
        m, n = Xtrain.shape
        # 创建一个矩阵
        new_Train_X = np.empty((m, n + 1))
        # 第一列填充1
        new_Train_X[:, 0] = 1
        # 后面使用源数据集补上
        new_Train_X[:, 1:] = Xtrain
        return new_Train_X  # 返回此新的数据集

    # 方法②
    def _gradient_descent(self, w, Ytrain, Xtrain):
        # 这是计算loss的时候的y的个数
        m = Ytrain.size
        # 加入有要求 差值（阈值）
        if self.diff is not None:
            loss_last = np.inf  # 这是一个非常大的数

        # self.loss_list = []  # 用来存放loss 方便观察  需要时用

        for step in range(self.n_iter):
            # 计算y_hat
            y_hat = 1. / (1. + np.exp(-np.dot(Xtrain, self.w)))  # 先得到z，然后扔进sigmoid函数求值
            # y_hat = self._predict_proba(Xtrain, self.w)

            # 计算梯度
            dJ = np.dot(y_hat - Ytrain, Xtrain) / Ytrain.size

            # 计算损失（对数损失函数）首先最外层会有一个-1/m(m=y.size)然后是对对数似然函数的计算，再相乘
            loss_new = -1 / m * np.sum(Ytrain * np.log(y_hat) + (1 - Ytrain) * np.log(1 - y_hat))

            # 这是书籍里的代码原函数，看不懂和公式的关系。
            # loss_new = -np.sum(np.log(y_hat * (2 * Ytrain - 1) + (1 - Ytrain))) / m

            # 记录损失
            # self.loss_list.append(loss_new)  # 需要时用
            # 显示最新一次损失
            print(f">>第{step}次梯度下降,损失:{loss_new}")

            # 达到阈值就停止迭代
            if self.diff is not None:
                if np.abs(loss_new - loss_last) <= self.diff:
                    break
                loss_last = loss_new

            # 更新w
            self.w -= self.eta * dJ

    def _Predict(self, X):
        # 预处理
        X = self._X_preprocess(X)
        # 预测y=1的概率
        y_pred = 1 / (1 + np.exp(-np.dot(X, self.w)))
        # 根据概率 预测类别
        return np.where(y_pred >= 0.5, 1, 0)



def test(X, y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_std = ss.transform(X_train)
    X_test_std = ss.transform(X_test)
    Clf = LogisticRegression(n_iter=200000, eta=0.01, diff=0.000001)
    Clf._Fit(X_train_std, Y_train)
    y_pred = Clf._Predict(X_test_std)
    accuray = accuracy_score(Y_test, y_pred)
    return accuray


if __name__ == '__main__':
    # 文件本身是(m,n) m是样本个数，n是特征值  方向摆放
    train_x = np.genfromtxt('Dataset/wine.data', delimiter=',', usecols=range(1, 14))
    train_y = np.genfromtxt('Dataset/wine.data', delimiter=',', usecols=0)
    idx = train_y != 3
    train_y = train_y[idx] - 1
    train_x = train_x[idx]
    accuray_mean = np.mean([test(train_x, train_y) for _ in range(3)])
    print(accuray_mean)