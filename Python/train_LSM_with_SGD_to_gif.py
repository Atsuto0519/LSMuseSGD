import numpy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import likely_chainer.models
import likely_chainer.optimizers


# シード値固定
numpy.random.seed(seed=10)

# モデル定義
dim = 2
alpha = 0.02
epoch = 100
model = likely_chainer.models.LSM(dimension=dim, learning_rate=alpha)
optimizer = likely_chainer.optimizers.SGD()
optimizer.setup(model)

# とりあえずランダムな重みのときをプロットしてみる
plot_x = numpy.linspace(-5, 10)
plot_y = model(plot_x).data

# とりあえずデータを用意する
train_x = [1, 2, 3]
train_y = [3, 2, 0]


# 簡単に学習させてみる
# エポックで回してみる
fig, ax = plt.subplots()
artists = []
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

    ax.set_ylim(-5, 5)
    pred_y = model(plot_x).data

    # 動的にアニメーション化
    if (ep + 1 == epoch):
        # 最後だけラベルを振る
        ax.scatter(train_x, train_y, color="RED", label="data")
        ax.plot(plot_x, plot_y, color="BLUE", label="initial line")
        line = ax.plot(plot_x, pred_y, color="GREEN", label="optimized line")
    else:
        ax.scatter(train_x, train_y, color="RED")
        ax.plot(plot_x, plot_y, color="BLUE")
        line = ax.plot(plot_x, pred_y, color="GREEN")
        title = ax.text(
                0.5, 1.01,
                "learning rate = {}, {} epoch score".format(alpha, ep + 1),
                transform=ax.transAxes, ha="center")

    artists.append(line + [title])


# GIFで保存
ax.legend()
anime = animation.ArtistAnimation(fig, artists, interval=100)
anime.save('images/anim.gif', writer="imagemagick")
