import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils
import pandas as pd

_TOP_PREDICTIONS_IN_OUTPUT = 20

class ModelExporter(object):

  def __init__(self, frame_features, model, reader):
    self.frame_features = frame_features
    self.model = model
    self.reader = reader

    with tf.Graph().as_default() as graph:
      self.inputs, self.outputs = self.build_inputs_and_outputs()
      self.graph = graph
      self.saver = tf.train.Saver(tf.trainable_variables(), sharded=True)

  def export_model(self, model_dir, global_step_val, last_checkpoint):
    """Exports the model so that it can used for batch predictions."""

    with self.graph.as_default():
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        self.saver.restore(session, last_checkpoint)

        signature = signature_def_utils.build_signature_def(
            inputs=self.inputs,
            outputs=self.outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: 
                         signature}

        model_builder = saved_model_builder.SavedModelBuilder(model_dir)
        model_builder.add_meta_graph_and_variables(session,
            tags=[tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save()

  def parse_csv(line):
        record_defaults = [[1.0] for col in range(20783)]

        parsed_line=tf.decode_csv(line,record_defaults)
        label=tf.cast(tf.reshape(parsed_line[:101],shape=(101,)),tf.int32)
        model_input_raw=tf.reshape(parsed_line[101:20581],shape=(20480,))
        model1_input=tf.reshape(parsed_line[20581:20682],shape=(101,))
        model2_input=tf.reshape(parsed_line[20682:],shape=(101,))
        d=label,model_input_raw,model1_input,model2_input
        return d


  def build_inputs_and_outputs(self):

    if self.frame_features:

      serialized_examples = tf.placeholder(tf.string, shape=(None,))

      fn = lambda x: self.build_prediction_graph(x)
      video_id_output, top_indices_output, top_predictions_output = (
          tf.map_fn(fn, serialized_examples, 
                    dtype=(tf.string, tf.int32, tf.float32)))

    else:

      serialized_examples = tf.placeholder(tf.string, shape=(None,))

      video_id_output, top_indices_output, top_predictions_output = (
          self.build_prediction_graph(serialized_examples))

    inputs = {"example_bytes": 
              saved_model_utils.build_tensor_info(serialized_examples)}

    outputs = {
        "video_id": saved_model_utils.build_tensor_info(video_id_output),
        "class_indexes": saved_model_utils.build_tensor_info(top_indices_output),
        "predictions": saved_model_utils.build_tensor_info(top_predictions_output)}

    return inputs, outputs



  def build_prediction_graph(self, serialized_examples):  


      

    all_feature=pd.read_csv("E:/Moe_Junyue/output.csv",header=None)
    train_input=tf.data.TextLineDataset(model1_path)
    train_dataset=train_input.map(parse_csv)


    with tf.name_scope("model"):
      result = self.model.create_model(
          model_input,
          model1_input=model1_input,
          model2_input=model2_input,
          num_frames=20,
          vocab_size=101,
          labels=labels_batch,
          is_training=False)

      for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)

      predictions = result["predictions"]

      top_predictions, top_indices = tf.nn.top_k(predictions, 
          _TOP_PREDICTIONS_IN_OUTPUT)
    return video_id, top_indices, top_predictions