import numpy


class SGD():
    """
    chainerのSGD風に使える最適化関数．
    """
    def __init__(self):
        self.optimizer = None

    def setup(self, model):
        if self.optimizer is None:
            self.model = model
        else:
            print("Please set model.")

    def update(self):
        self.model.w += self.model.learning_rate * self.model.grads


class LeastSquaresMethod():
    """
    chainerのモデル風に使えるモデル．
    """
    def __init__(self, *, dimension=2, learning_rate=0.1):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.w = numpy.random.randn(self.dimension + 1)
        self.grads = numpy.zeros([self.dimension + 1])

    def __call__(self, *args):
        # パラメータが多すぎたらエラー
        if (len(args) > 2):
            print("Please check parameter.")
        elif (len(args) > 0):
            # ただのスコア計算なら
            self.x = numpy.array(args[0])
            self.data = self.score()
            # 学習するなら
            if (len(args) > 1):
                self.y = numpy.array(args[1])
                self.error = (self.data - self.y)

        return self

    def score(self):
        """
        データ点を入れたときのyの推定値．
        """
        self.X = numpy.array([(x ** numpy.ones([self.dimension + 1]))[::-1]
                              for x in self.x])
        self.X = self.X ** numpy.arange(self.dimension + 1)[::-1]
        scores = numpy.dot(self.X, self.w)

        return scores

    def zerograds(self):
        self.grads = numpy.zeros([self.dimension])

    def backward(self):
        self.grads = - numpy.dot(self.error, self.X)
