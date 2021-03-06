import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import likely_chainer.models
import likely_chainer.optimizers


# シード値固定
numpy.random.seed(seed=10)

# モデル定義
dim = 1
alpha = 0.1
model = likely_chainer.models.LSM(
    dimension=dim,
    learning_rate=alpha,
    define_by_run=True)
optimizer = likely_chainer.optimizers.SGD()
optimizer.setup(model)

# とりあえずランダムな重みのときをプロットしてみる
plot_x = numpy.linspace(-5, 10)
plot_y = model(plot_x).data
plt.plot(plot_x, plot_y, label="initial score")

# とりあえずデータを用意する
train_x = [1, 2, 3]
train_y = [3, 2, 0]
plt.scatter(train_x, train_y, color="RED")

# 簡単に学習させてみる
# エポックで回してみる
epoch = 1000
for ep in range(epoch):
    # まずはデータをシャッフル
    perm = numpy.random.permutation(len(train_x))
    train_x = numpy.array(train_x)[perm]
    train_y = numpy.array(train_y)[perm]

    for (x, y) in zip(train_x, train_y):
        # 完璧なChainer風に仕上げた
        model.zerograds()
        loss = model([x], [y])
        loss.backward()
        optimizer.update()

# 最終結果をプロット
plot_y = model(plot_x).data
plt.plot(plot_x, plot_y,
         label=str(ep+1) + " epoch score",
         color=cm.gray(1-ep/epoch))

# 表示
plt.ylim([-5, 5])
plt.legend()
plt.title("learning rate = "+str(alpha))
plt.savefig("images/dim"+str(dim)+".png")
