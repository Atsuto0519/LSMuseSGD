import numpy
import chainer
from chainer import Variable
import chainer.functions as F


def func_y(w, x, dim):
    pred_y = sum([w[d] * (x ** d) for d in range(dim + 1)])
    return pred_y.reshape(pred_y.shape[0])


def func_J(y, pred_y):
    return 0.5 * F.sqrt(F.mean_squared_error(y, pred_y))


class LSM():
    """
    chainerのモデル風に使えるモデル．
    """
    def __init__(self, *, dimension=2, learning_rate=0.1, define_by_run=False):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.define_by_run = define_by_run

        if self.define_by_run:
            self.w = numpy.random.randn(self.dimension + 1)
            self.w = self.w.astype(numpy.float32)
            self.w = Variable(self.w.reshape(self.dimension + 1))
            self.w.cleargrad()
            if self.w.grad is None:
                self.grads = numpy.zeros([self.dimension + 1])
            else:
                self.grads = self.w.grad.reshape(self.dimension + 1)
        else:
            self.w = numpy.random.randn(self.dimension + 1)
            self.grads = numpy.zeros([self.dimension + 1])

    def __call__(self, *args):
        # パラメータが多すぎたらエラー
        if (len(args) > 2):
            print("Please check parameter.")
        elif (len(args) > 0):
            # ただのスコア計算なら
            self.x = numpy.array(args[0])
            self.x = self.x.astype(numpy.float32)
            self.data = self.__score__()
            if self.define_by_run:
                pred_y = self.data
                self.data = self.data.data.reshape(self.data.data.shape[0])

            # 学習するなら
            if (len(args) > 1):
                self.y = numpy.array(args[1])
                self.y = self.y.astype(numpy.float32)
                if self.define_by_run:
                    self.J = func_J(Variable(self.y), pred_y)
                else:
                    self.error = (self.y - self.data)

        return self

    def __score__(self):
        """
        データ点を入れたときのyの推定値．
        """
        if self.define_by_run:
            scores = func_y(self.w, Variable(self.x), self.dimension)
        else:
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

        if self.define_by_run:
            self.w.cleargrad()
            if self.w.grad is None:
                self.grads = numpy.zeros([self.dimension + 1])
            else:
                self.grads = self.w.grad.reshape(self.dimension + 1)
        else:
            self.grads = numpy.zeros([self.dimension + 1])

    def backward(self):
        if self.define_by_run:
            self.J.backward(retain_grad=True)
            self.grads = - self.w.grad.reshape(self.dimension + 1)
        else:
            self.grads = numpy.dot(self.error, self.X)
