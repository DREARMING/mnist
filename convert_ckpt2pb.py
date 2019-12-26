# coding utf-8

from tensorflow.python.framework import graph_util
import tensorflow as tf

output_node_names = "number"

save_pb_model_path = "module/mnist.pb"

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('module/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint("module/"))

    output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=sess.graph_def,  # 等于:sess.graph_def
        output_node_names=[output_node_names])  # 如果有多个输出节点，以逗号隔开

    with tf.gfile.GFile(save_pb_model_path, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出