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
  flags.DEFINE_string("train_dir", "E:/Moe_Junyue/moe_model/3moe",
                      "The directory to load the model files from.")
  flags.DEFINE_string("output_file", "E:/Moe_Junyue/dataset/predictions_test_3moe.csv",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "E:/UCF101_TRAIN/ucf_train.tfrecord",
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
  flags.DEFINE_integer("top_k", 101,
                       "How many predictions to output per video.")
  flags.DEFINE_integer("check_point",-1,
                       "Model checkpoint to load, -1 for latest.")

def format_lines( video_ids,predictions, top_k,feature_val,label_val):
  batch_size = len(video_ids)
  for video_index in range(batch_size):

    line = [( predictions[video_index][class_index])
            for class_index in range(top_k)]
    line2 = [( feature_val[video_index][feature_index])
            for feature_index in range(1024)]
    line3 = [( label_val[video_index][label_index])
            for label_index in range(101)]
    yield ",".join(" %f" % pair for pair in line3) +","+",".join(" %f" % pair for pair in line2)+","+",".join(" %f" % pair for pair in line)+"\n"


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch = True,
                            enqueue_many=True))
    return video_id_batch, video_batch, unused_labels, num_frames_batch

def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
  with tf.Session() as sess, gfile.Open(out_file_location, "w+") as out_file:
    video_id_batch, video_batch, labels_batch, num_frames_batch = get_input_data_tensors(reader, data_pattern, batch_size)
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
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]
    feature_input = tf.get_collection("model_input1")[0]

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
    sum_hit_at_one=0
    try:
      while not coord.should_stop():
          video_id_batch_val, video_batch_val,label_val, num_frames_batch_val = sess.run([video_id_batch, video_batch, labels_batch, num_frames_batch])
          predictions_val,feature_val = sess.run([predictions_tensor,feature_input], feed_dict={input_tensor: video_batch_val, num_frames_tensor: num_frames_batch_val})
          now = time.time()
          hit_at_one = eval_util.calculate_hit_at_one(predictions_val,label_val)
          num_examples_processed += len(video_batch_val)
          num_classes = predictions_val.shape[1]
          sum_hit_at_one += hit_at_one * len(video_batch_val)
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
          for line in format_lines(video_id_batch_val, predictions_val, top_k,feature_val,label_val):
            out_file.write(line)
          out_file.flush()


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
