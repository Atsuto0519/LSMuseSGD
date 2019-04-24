import numpy


class LSM():
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
            self.data = self.__score__()
            # 学習するなら
            if (len(args) > 1):
                self.y = numpy.array(args[1])
                self.error = (self.data - self.y)

        return self

    def __score__(self):
        """
        データ点を入れたときのyの推定値．
        """
        self.X = numpy.array([(x ** numpy.ones([self.dimension + 1]))
                              for x in self.x])
        self.X = self.X ** numpy.arange(self.dimension + 1)
        scores = numpy.dot(self.X, self.w)

        return scores

    def zerograds(self):
        attr_self = [i for i in dir(self) if "__" not in i]
        if "x" in attr_self:
            del self.x
        if "y" in attr_self:
            del self.y
        if "X" in attr_self:
            del self.X
        if "data" in attr_self:
            del self.data
        if "error" in attr_self:
            del self.error

        self.grads = numpy.zeros([self.dimension])

    def backward(self):
        self.grads = - numpy.dot(self.error, self.X)
