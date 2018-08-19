# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import create_data as cd
import forward as fd

STEP = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.9
REGULARIZER = 0.01


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_c = cd.generateds()

    y = fd.forward(x, REGULARIZER)

    # 定义损失函数
    loss_mes = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mes + tf.add_n(tf.get_collection('losses'))

    # 定义指数衰减学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        300 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义训练工程
    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEP):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})

            if i % 2000 == 0:
                loss = sess.run(loss_mes, feed_dict={x: X, y_: Y_})
                print("after %d steps, loss is : %f" % (i, loss))

        #产生网格坐标
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        predict = sess.run(y, feed_dict={x: grid})
        predict = predict.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, predict, levels=[0.5])
    plt.show()


if __name__ == '__main__':
    backward()
