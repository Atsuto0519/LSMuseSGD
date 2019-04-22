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
