from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.framework import graph_util

# 读取数据集
mnist = input_data.read_data_sets("data/", one_hot=True)

output_node_names = "number"

save_pb_model_path = "module/mnist.pb"
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(save_pb_model_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
    with tf.Session() as sess:
        # 通过图去获取 操作变量
        graph = sess.graph
        inputImage = graph.get_tensor_by_name("inputImage:0")
        pre_number = graph.get_tensor_by_name("number:0")
        # 用测试数据集的第一张照片做预测
        pic = np.array([mnist.test.images[0]])
        startTime = time.time()
        print(sess.run(pre_number, feed_dict={inputImage:pic}))
        # 大约耗时 15.6 ms
        print("耗时：%.3f second" % (time.time() - startTime))