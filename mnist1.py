# coding utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# 读取数据集
mnist = input_data.read_data_sets("data/", one_hot=True)

# 定义一个占位符x，代表输入的测试图片，规格是 28x28，用二维数组表示多张图片，其中第二维度是图片像素平面化之后的数据表示，所以是784个长度
x = tf.placeholder(tf.float32, [None, 784], name="inputImage")  # 张量的形状是[None, 784]，None表第一个维度任意

# 定义变量W,b,是可以被修改的张量，用来存放机器学习模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现模型, y是预测分布
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型，y_是实际分布
y_ = tf.placeholder("float", [None, 10], name="inputLabel")
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 交叉嫡，cost function

# 使用梯度下降来降低cost，学习速率为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在一个Session中启动模型，并初始化变量
with tf.Session() as sess:
    # 初始化已经创建的变量
    init = tf.global_variables_initializer()
    # 创建 saver 用来保存训练好的模型，后面不用重复进行训练，节省时间
    saver = tf.train.Saver()
    sess.run(init)
    # 训练模型，运行1000次，每次随机抽取100个
    for i in range(1, 1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 保存单张图片的预测输出，并且输入其值
    # 下面函数的意义是， 从 y 的 dimension 为 1 这个数组中获取里面元素最大的index返回，这里指定返回的类型是 int32，跟java的int同一个类型，并且，指定操作名为 number
    # y的 shape 是 [none, 10], 所以 输入1 获取 10个label中，概率最大的index 返回
    preNum = tf.arg_max(y, 1, output_type='int32', name="number")
    # 取测试集中第一张图片进行测试
    pic = np.array([mnist.test.images[0]])
    # 运行测试代码, 返回 int[] 数组
    print(sess.run(preNum, feed_dict={x: pic}))

    # 保存模型
    savePath = saver.save(sess, "module/model")

    # 测试代码
    # 验证正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print('save path is : %s' % savePath)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



