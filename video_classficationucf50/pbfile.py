import os
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pb_path = 'E:/ucf50_train/yt8m2/v2/models/frame/sample_model/export/step_10/saved_model.pb'

run_meta = tf.RunMetadata()
with tf.Graph().as_default():
    output_graph_def = graph_pb2.GraphDef()
    with open(pb_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(output_graph_def, name="")
        print('model loaded!')
    all_keys = sorted([n.name for n in tf.get_default_graph().as_graph_def().node])
    # for k in all_keys:
    #   print(k)

    with tf.Session() as sess:
        flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta,
            options=tf.profiler.ProfileOptionBuilder.float_operation())
        print("test flops:{:,}".format(flops.total_float_ops))