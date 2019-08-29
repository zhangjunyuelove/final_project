import os
import time

import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("train_dir", "E:/Moe_Junyue/moe_model/three_models",
                      "The directory to load the model files from.")
  flags.DEFINE_string("output_file", "predictions_train_3models.csv",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern",  "E:/Moe_Junyue/dataset/3combined_test.tfrecords",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", True,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_integer(
      "batch_size", 256,
      "How many examples to process per batch.")
  flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")


  # Other flags.
  flags.DEFINE_integer("num_readers", 4,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 1,
                       "How many predictions to output per video.")
  flags.DEFINE_integer("check_point",-1,
                       "Model checkpoint to load, -1 for latest.")



def decode(serialized_example):
  # NOTE: You might get an error here, because it seems unlikely that the features
  # called 'coord2d' and 'coord3d', and produced using `ndarray.flatten()`, will
  # have a scalar shape. You might need to change the shape passed to
  # `tf.FixedLenFeature()`.
  features = tf.parse_single_example(
      serialized_example,
      features={'labels_batch': tf.FixedLenFeature([101], tf.float32),
                'model_input_raw' : tf.FixedLenFeature([1024], tf.float32),
                'model1_input' : tf.FixedLenFeature([101], tf.float32),
                'model2_input' : tf.FixedLenFeature([101], tf.float32),
                'model3_input' : tf.FixedLenFeature([101], tf.float32),})
    

  # NOTE: No need to cast these features, as they are already `tf.float32` values.
  return features['labels_batch'], features['model_input_raw'],features['model1_input'], features['model2_input'],features['model3_input']


def get_input_data_tensors(data_pattern,batch_size):
    with tf.name_scope("input"):
       filename = [data_pattern]
       dataset = tf.data.TFRecordDataset(filename).map(decode)
       train_dataset_epoch = dataset.repeat(1)
       batch_model_1=train_dataset_epoch.batch(batch_size)
       iterator = batch_model_1.make_one_shot_iterator()
       labels_batch,model_input_raw,model1_input,model2_input,model3_input = iterator.get_next()
       return labels_batch, model_input_raw, model1_input, model2_input,model3_input




def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
  with tf.Session() as sess, gfile.Open(out_file_location, "w+") as out_file:
    labels_batch, model_input_raw, model1_input, model2_input,model3_input = get_input_data_tensors( data_pattern, batch_size)
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      if FLAGS.check_point < 0:
        meta_graph_location = latest_checkpoint + ".meta"
      else:
        meta_graph_location = FLAGS.train_dir + "/model.ckpt-" + str(FLAGS.check_point) + ".meta"
        latest_checkpoint = FLAGS.train_dir + "/model.ckpt-" + str(FLAGS.check_point)
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
    input_tensor = tf.get_collection("input_batch_raw")[0]
    input_tensor1 = tf.get_collection("input_batch1")[0]
    input_tensor2 = tf.get_collection("input_batch2")[0]
    input_tensor3 = tf.get_collection("input_batch3")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    out_file.write("VideoId,LabelConfidencePairs\n")
    sum_hit_at_one=0
    try:
      while not coord.should_stop():
          labels_batch_val, model_input_raw_val, model1_input_val, model2_input_val, model3_input_val = sess.run([labels_batch, model_input_raw, model1_input, model2_input,model3_input])
          predictions_val, = sess.run([predictions_tensor], feed_dict={input_tensor: model_input_raw_val, input_tensor1: model1_input_val,input_tensor2: model2_input_val,input_tensor3: model3_input_val})
          now = time.time()
          print(model1_input_val.shape[0])
          hit_at_one = eval_util.calculate_hit_at_one(predictions_val,labels_batch_val)


          num_examples_processed += predictions_val.shape[0]

          sum_hit_at_one += hit_at_one * len(model_input_raw_val)
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))


    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
        accuracy_performance=sum_hit_at_one / num_examples_processed
        logging.info('------------------final performance----------------'+("%.4f" % accuracy_performance))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
    FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()