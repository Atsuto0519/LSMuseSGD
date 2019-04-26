# LSMuseSGD [![Build Status](https://travis-ci.com/Atsuto0519/LSMuseSGD.svg?branch=master)](https://travis-ci.com/Atsuto0519/LSMuseSGD)

確率的勾配降下法(SGD)を使って最小二乗法を実装した．


# Usage

## Python

PythonはChainerによく似たライブラリを用意したので[LSM_likely_Chainer.py](./Python/LSM_likely_Chainer.py)を利用されたい．

また，適当にデータを用意して実行したものが[train_LSM_with_SGD.py](./Python/train_LSM_with_SGD.py)をサンプルとして用意した．

### Define model and training

```Python
import likely_chainer.models
import likely_chainer.optimizers

# Define parameter
dim = 2
alpha = 0.01
epoch = 1000
model = likely_chainer.models.LSM(dimension=dim, learning_rate=alpha)
# If you want define by run ...
model = likely_chainer.models.LSM(dimension=dim, learning_rate=alpha, define_by_run=True)

optimizer = likely_chainer.optimizers.SGD()
optimizer.setup(model)

# This looks like chainer ...
for ep in range(epoch):
  for (x, y) in zip(train_x, train_y):
      model.zerograds()
      loss = model(x, y)
      loss.backward()
      optimizer.update()

# Predict data from test_x
pred_y = model(test_x).data
```
