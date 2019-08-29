import tensorflow as tf
import pandas

csv = pandas.read_csv("E:/Moe_Junyue/dataset/test_combined.csv",header=None).values
with tf.python_io.TFRecordWriter("E:/Moe_Junyue/dataset/combined_test.tfrecords") as writer:
    for row in csv:
        label, feature1, feature2, feature3 = row[:101],row[101:1125],row[1125:1226],row[1226:1327]

        example = tf.train.Example()
        example.features.feature["labels_batch"].float_list.value.extend(label)
        example.features.feature["model_input_raw"].float_list.value.extend(feature1)
        example.features.feature["model1_input"].float_list.value.extend(feature2)
        example.features.feature["model2_input"].float_list.value.extend(feature3)


        writer.write(example.SerializeToString())