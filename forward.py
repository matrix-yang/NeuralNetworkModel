# code:utf-8
import tensorflow as tf


# 产生一组权重值，并把权重加入losses的
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 产生一个偏置值
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=shape))
    return b


'''构建网络计算前项传播的结果
    输入层节点 2    
    隐含层节点 11
    输出层节点 1
'''
def forward(x, regularizer):
    w1 = get_weight((2, 11), regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight((11, 1), regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2

    return y
