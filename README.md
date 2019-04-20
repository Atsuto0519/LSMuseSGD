# LSMuseSGD [![Build Status](https://travis-ci.com/Atsuto0519/LSMuseSGD.svg?branch=master)](https://travis-ci.com/Atsuto0519/LSMuseSGD)

確率的勾配降下法(SGD)を使って最小二乗法を実装した．


# Usage

## Python

PythonはChainerによく似たライブラリを用意したので[LSM_likely_Chainer.py](./Python/LSM_likely_Chainer.py)を利用されたい．

また，適当にデータを用意して実行したものが[train_LSM_with_SGD.py](./Python/train_LSM_with_SGD.py)をサンプルとして用意した．

### Define model and training

```Python
from LSM_likely_Chainer import LeastSquaresMethod, SGD

dim = n
alpha = 0.01
model = LeastSquaresMethod(dimension=dim, learning_rate=alpha)
optimizer = SGD()
optimizer.setup(model)

for (x, y) in zip(train_x, train_y):
    model.zerograds()
    loss = model([x], [y])
    loss.backward()
    optimizer.update()
```
