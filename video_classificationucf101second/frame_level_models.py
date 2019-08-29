# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

import scipy.io as sio
import numpy as np

FLAGS = flags.FLAGS


flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")



flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                    "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")
flags.DEFINE_integer("num_model", 2,
                     "number of models for moe")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')

flags.DEFINE_integer("netvlad_cluster_size", 32,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                   "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                   "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                   "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")


flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_float(
    "moe_l2", 1e-8,
    "L2 penalty for MoeModel.")



flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                     "ReLU after the NetFV hidden layer.")


flags.DEFINE_bool("fv_couple_weights", True,
                     "Coupling cluster weights or not")
 
flags.DEFINE_float("fv_coupling_factor", 0.01,
                     "Coupling factor")


flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "LogisticModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")



flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")

flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                     "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                     "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")


flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")
class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

    feature_size = model_input.get_shape().as_list()[2]
    model_input1 = utils.SampleRandomFrames(model_input, num_frames,
                                             20)
    reshaped_input = tf.reshape(model_input1, [-1, feature_size*20])
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output, 'feature':reshaped_input}


class LightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad


class NetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        

        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad


class NetVLAGD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        gate_weights = tf.get_variable("gate_weights",
            [1, self.cluster_size,self.feature_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        gate_weights = tf.sigmoid(gate_weights)

        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])

        vlagd = tf.matmul(activation,reshaped_input)
        vlagd = tf.multiply(vlagd,gate_weights)

        vlagd = tf.transpose(vlagd,perm=[0,2,1])
        
        vlagd = tf.nn.l2_normalize(vlagd,1)

        vlagd = tf.reshape(vlagd,[-1,self.cluster_size*self.feature_size])
        vlagd = tf.nn.l2_normalize(vlagd,1)

        return vlagd


class NetFV():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
     
        covar_weights = tf.get_variable("covar_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(mean=1.0, stddev=1 /math.sqrt(self.feature_size)))
      
        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights,eps)

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [self.cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        if not FLAGS.fv_couple_weights:
            cluster_weights2 = tf.get_variable("cluster_weights2",
              [1,self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor,cluster_weights)

        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        fv1 = tf.matmul(activation,reshaped_input)
        
        fv1 = tf.transpose(fv1,perm=[0,2,1])

        # computing second order FV
        a2 = tf.multiply(a_sum,tf.square(cluster_weights2)) 

        b2 = tf.multiply(fv1,cluster_weights2) 
        fv2 = tf.matmul(activation,tf.square(reshaped_input)) 
     
        fv2 = tf.transpose(fv2,perm=[0,2,1])
        fv2 = tf.add_n([a2,fv2,tf.scalar_mul(-2,b2)])

        fv2 = tf.divide(fv2,tf.square(covar_weights))
        fv2 = tf.subtract(fv2,a_sum)

        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
      
        fv2 = tf.nn.l2_normalize(fv2,1)
        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2,1)

        fv1 = tf.subtract(fv1,a)
        fv1 = tf.divide(fv1,covar_weights) 

        fv1 = tf.nn.l2_normalize(fv1,1)
        fv1 = tf.reshape(fv1,[-1,self.cluster_size*self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1,1)

        return tf.concat([fv1,fv2],1)








class NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input2 = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input2 = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    
    max_frames = model_input2.get_shape().as_list()[1]
    feature_size = model_input2.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input2, [-1, feature_size])
    
    model_input1 = utils.SampleRandomFrames(model_input, num_frames,
                                             20)
    reshaped_input1 = tf.reshape(model_input1, [-1, feature_size*20])
    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)


  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 



    vlad = vlad_video

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    prediction_final=aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    return {"predictions": prediction_final, "feature": reshaped_input1}
  


class GruModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of GRUs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    gru_size = FLAGS.gru_cells
    number_of_layers = FLAGS.gru_layers
    backward = FLAGS.gru_backward
    random_frames = FLAGS.gru_random_sequence
    iterations = FLAGS.iterations
    
    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)
 
    if backward:
        model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1) 
    
    stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

class GatedMoe(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   num_model=2,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    num_model=num_model or FLAGS.num_model
    

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    hidden_size_1=max_frames*feature_size
    reshape_input1=tf.reshape(model_input, [-1, max_frames*feature_size])
    
    moegating_weights = tf.get_variable("moegating_weights_1",
        [hidden_size_1, num_model],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden_size_1)))
     
    moe_gates = tf.matmul(reshape_input1, moegating_weights)    

    if add_batch_norm:
      moe_gates = slim.batch_norm(
          moe_gates,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn1")

    else:
      hidden1_biases1 = tf.get_variable("hidden1_biases1",
        [hidden_size_1],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases1", hidden1_biases1)
      moe_gates += hidden1_biases1
      
      
    moe_gates_softmax=tf.nn.softmax(moe_gates)
    moe_gates_model1=moe_gates_softmax[:,0]
    moe_gates_model2=moe_gates_softmax[:,1]
    moe_gates_model11=tf.cast(tf.expand_dims(moe_gates_model1, 1), tf.float32)
    moe_gates_model21=tf.cast(tf.expand_dims(moe_gates_model2, 1), tf.float32)
    moe_gates_modelnew1=tf.tile(moe_gates_model11,[1,vocab_size])
    moe_gates_modelnew2=tf.tile(moe_gates_model21,[1,vocab_size])

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    prediction_model1=aggregated_model().create_model(
        model_input=avg_pooled,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight1=tf.multiply(prediction_model1,moe_gates_modelnew1)





    if random_frames:
      model_input_new = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input_new = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    max_frames = model_input_new.get_shape().as_list()[1]
    feature_size = model_input_new.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input_new, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)


  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 



    vlad = vlad_video

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    prediction_model2=aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight2=tf.multiply(prediction_model2,moe_gates_modelnew2)    
    prediction_final=tf.add(prediction_model_weight1,prediction_model_weight2)
    return {"predictions": prediction_final}


class GatedMoe2(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   num_model=2,
                   l2_penalty=1e-8,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    num_model=num_model or FLAGS.num_model
    l2_penalty = FLAGS.moe_l2;
    

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]

    reshape_input1=tf.reshape(model_input, [-1, max_frames*feature_size])
    
    moegate_activations = slim.fully_connected(
        reshape_input1,
        vocab_size * (num_model + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="moegates")
      
      
    mooegating_distribution = tf.nn.softmax(tf.reshape(
        moegate_activations,
        [-1, num_model + 1]))  # (Batch * #Labels) x (num_model + 1)
    moe_gates_model1=tf.reshape(mooegating_distribution[:,0],[-1,vocab_size])
    moe_gates_model2=tf.reshape(mooegating_distribution[:,1],[-1,vocab_size])


    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators
                               
    reshaped_input1=avg_pooled

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    prediction_model1=aggregated_model().create_model(
        model_input=avg_pooled,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight1=tf.multiply(prediction_model1,moe_gates_model1)





    if random_frames:
      model_input_new = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input_new = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    max_frames = model_input_new.get_shape().as_list()[1]
    feature_size = model_input_new.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input_new, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)


  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 



    vlad = vlad_video

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    prediction_model2=aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight2=tf.multiply(prediction_model2,moe_gates_model2)    
    prediction_final=tf.add(prediction_model_weight1,prediction_model_weight2)
    return {"predictions": prediction_final, "feature": reshaped_input1}


class GatedMoe3m(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   num_model=2,
                   l2_penalty=1e-8,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    num_model=num_model or FLAGS.num_model
    l2_penalty = FLAGS.moe_l2;
    

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]

    reshape_input1=tf.reshape(model_input, [-1, max_frames*feature_size])
    
    moegate_activations = slim.fully_connected(
        reshape_input1,
        8192,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="moegates")
    drop_out=tf.nn.dropout(moegate_activations,0.8)
    moegate_activations1 = slim.fully_connected(
        drop_out,
        vocab_size * (num_model + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="moegates2")
    drop_out2=tf.nn.dropout(moegate_activations1,0.8)      
      
    mooegating_distribution = tf.nn.softmax(tf.reshape(
        drop_out2,
        [-1, num_model + 1]))  # (Batch * #Labels) x (num_model + 1)
    moe_gates_model1=tf.reshape(mooegating_distribution[:,0],[-1,vocab_size])
    moe_gates_model2=tf.reshape(mooegating_distribution[:,1],[-1,vocab_size])
    moe_gates_model3=tf.reshape(mooegating_distribution[:,2],[-1,vocab_size])

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators
                               
    reshaped_input1=avg_pooled

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    prediction_model1=aggregated_model().create_model(
        model_input=avg_pooled,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight1=tf.multiply(prediction_model1,moe_gates_model1)





    if random_frames:
      model_input_new = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input_new = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    max_frames = model_input_new.get_shape().as_list()[1]
    feature_size = model_input_new.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input_new, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)


  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 



    vlad = vlad_video

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    prediction_model2=aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight2=tf.multiply(prediction_model2,moe_gates_model2)
    
    
    video_NetFV = NetFV(1024,max_frames,cluster_size, add_batch_norm, is_training)





    with tf.variable_scope("video_FV"):
        fv_video = video_NetFV.forward(reshaped_input[:,0:1024]) 



    fv = fv_video

    fv_dim = fv.get_shape().as_list()[1] 
    hidden3_weights = tf.get_variable("hidden3_weights",
      [fv_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    
    activation1 = tf.matmul(fv, hidden3_weights)

    if add_batch_norm and relu:
      activation1 = slim.batch_norm(
          activation1,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden3_bn")
    else:
      hidden3_biases = tf.get_variable("hidden3_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden3_biases", hidden3_biases)
      activation1 += hidden3_biases
   
    if relu:
      activation1 = tf.nn.relu6(activation1)

    if gating:
        gating_weights3 = tf.get_variable("gating_weights_23",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation1, gating_weights3)
        
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn3")
        else:
          gating_biases3 = tf.get_variable("gating_biases3",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases3

        gates = tf.sigmoid(gates)

        activation1 = tf.multiply(activation1,gates)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    prediction_model3=aggregated_model().create_model(
        model_input=activation1,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    
    
    prediction_model_weight3=tf.multiply(prediction_model3,moe_gates_model3)    
    prediction_final1=tf.add(prediction_model_weight1,prediction_model_weight2)
    prediction_final=tf.add(prediction_final1,prediction_model_weight3)
    return {"predictions": prediction_final, "feature": reshaped_input1}


class GatedMoe_new(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   num_model=2,
                   l2_penalty=1e-8,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    num_model=num_model or FLAGS.num_model
    l2_penalty = FLAGS.moe_l2;
    

    max_frames1 = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]

    reshape_input1=tf.reshape(model_input, [-1, max_frames1*feature_size])
    

    moegate_activations1 = slim.fully_connected(
        reshape_input1,
        vocab_size * (num_model + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="moegates1")
          
      
    mooegating_distribution = tf.nn.softmax(tf.reshape(
        moegate_activations1,
        [-1, num_model + 1]))  # (Batch * #Labels) x (num_model + 1)
    moe_gates_model1=tf.reshape(mooegating_distribution[:,0],[-1,vocab_size])
    moe_gates_model2=tf.reshape(mooegating_distribution[:,1],[-1,vocab_size])
    moe_gates_model3=tf.reshape(mooegating_distribution[:,2],[-1,vocab_size])

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators
                               
    reshaped_input1=avg_pooled

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    prediction_model1=aggregated_model().create_model(
        model_input=avg_pooled,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight1=tf.multiply(prediction_model1,moe_gates_model1)





    if random_frames:
      model_input_new = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input_new = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    max_frames = model_input_new.get_shape().as_list()[1]
    feature_size = model_input_new.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input_new, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)

    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)


  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 



    vlad = vlad_video

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    prediction_model2=aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
    prediction_model_weight2=tf.multiply(prediction_model2,moe_gates_model2)

    reshape_input2=tf.reshape(model_input, [-1, max_frames1,feature_size])
    image_input=tf.reshape(reshape_input2[:,0,:],[-1,feature_size])
    image_activation = slim.fully_connected(
        image_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))    
    
    prediction_model_weight3=tf.multiply(image_activation,moe_gates_model3)    
    prediction_final1=tf.add(prediction_model_weight1,prediction_model_weight2)
    prediction_final=tf.add(prediction_final1,prediction_model_weight3)
    
    prediction_final=tf.add(prediction_model_weight1,prediction_model_weight2)
    return {"predictions": prediction_final, "feature": reshaped_input1}




class second_train(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   model1_input,
                   model2_input,
                   vocab_size,
                   num_frames,
                   add_batch_norm=None,
                   is_training=True,
                   num_model=2,
                   l2_penalty=1e-8,
                   **unused_params):
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    num_model=num_model or FLAGS.num_model

    


    reshape_input1=tf.reshape(model_input, [-1, 1024])


    

    
    moegating_weights = tf.get_variable("moegating_weights_1",
        [1024, num_model],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(1024)))
     
    moe_gates = tf.matmul(reshape_input1, moegating_weights)    

    if add_batch_norm:
      moe_gates = slim.batch_norm(
          moe_gates,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn1")

    else:
      hidden1_biases1 = tf.get_variable("hidden1_biases1",
        [1024],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases1", hidden1_biases1)
      moe_gates += hidden1_biases1
      
      
    moe_gates_softmax=tf.nn.softmax(moe_gates)
    moe_gates_model1=moe_gates_softmax[:,0]
    moe_gates_model2=moe_gates_softmax[:,1]
    moe_gates_model11=tf.cast(tf.expand_dims(moe_gates_model1, 1), tf.float32)
    moe_gates_model21=tf.cast(tf.expand_dims(moe_gates_model2, 1), tf.float32)
    moe_gates_modelnew1=tf.tile(moe_gates_model11,[1,vocab_size])
    moe_gates_modelnew2=tf.tile(moe_gates_model21,[1,vocab_size])
    
    
    
    model1_input=tf.cast(model1_input,tf.float32)
    model2_input=tf.cast(model2_input,tf.float32)
    prediction_model_weight1=tf.multiply(model1_input,moe_gates_modelnew1) 
    prediction_model_weight2=tf.multiply(model2_input,moe_gates_modelnew2)    
    prediction_1=tf.add(prediction_model_weight1,prediction_model_weight2)
    prediction_final=tf.nn.softmax(prediction_1)
    return {"predictions": prediction_final}






class second_train2(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   model1_input,
                   model2_input,
                   vocab_size,
                   num_frames,
                   add_batch_norm=None,
                   is_training=True,
                   num_model=2,
                   l2_penalty=1e-8,
                   **unused_params):
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    num_model=num_model or FLAGS.num_model

    


    reshape_input1=tf.reshape(model_input, [-1, 1024])


    

    
    moegate_activations = slim.fully_connected(
        reshape_input1,
        vocab_size * (num_model),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="moegates")
      
      
    mooegating_distribution = tf.nn.softmax(tf.reshape(
        moegate_activations,
        [-1, num_model]))  # (Batch * #Labels) x (num_model )
    moe_gates_model1=tf.reshape(mooegating_distribution[:,0],[-1,vocab_size])
    moe_gates_model2=tf.reshape(mooegating_distribution[:,1],[-1,vocab_size])
    
    
    
    model1_input=tf.cast(model1_input,tf.float32)
    model2_input=tf.cast(model2_input,tf.float32)
    prediction_model_weight1=tf.multiply(model1_input,moe_gates_model1) 
    prediction_model_weight2=tf.multiply(model2_input,moe_gates_model2)    
    prediction_1=tf.add(prediction_model_weight1,prediction_model_weight2)

    return {"predictions": prediction_1}







class second_train3(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   model1_input,
                   model2_input,
                   model3_input,
                   vocab_size,
                   num_frames,
                   add_batch_norm=None,
                   is_training=True,
                   num_model=2,
                   l2_penalty=1e-8,
                   **unused_params):
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    num_model=num_model or FLAGS.num_model

    


    reshape_input1=tf.reshape(model_input, [-1, 1024])


    

    
    moegate_activations = slim.fully_connected(
        reshape_input1,
        vocab_size * (num_model+1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="moegates")
      
      
    mooegating_distribution = tf.nn.softmax(tf.reshape(
        moegate_activations,
        [-1, num_model+1]))  # (Batch * #Labels) x (num_model + 1)
    moe_gates_model1=tf.reshape(mooegating_distribution[:,0],[-1,vocab_size])
    moe_gates_model2=tf.reshape(mooegating_distribution[:,1],[-1,vocab_size])
    moe_gates_model3=tf.reshape(mooegating_distribution[:,2],[-1,vocab_size])
    
    
    model1_input=tf.cast(model1_input,tf.float32)
    model2_input=tf.cast(model2_input,tf.float32)
    model3_input=tf.cast(model3_input,tf.float32)
    prediction_model_weight1=tf.multiply(model1_input,moe_gates_model1) 
    prediction_model_weight2=tf.multiply(model2_input,moe_gates_model2)   
    prediction_model_weight3=tf.multiply(model3_input,moe_gates_model3) 
    prediction_1=tf.add(prediction_model_weight1,prediction_model_weight2)
    prediction_2=tf.add(prediction_1,prediction_model_weight3)
    prediction_final=tf.nn.softmax(prediction_2)
    return {"predictions": prediction_final}



class second_train_ensemble(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   model1_input,
                   model2_input,
                   vocab_size,
                   num_frames,
                   add_batch_norm=None,
                   is_training=True,
                   num_model=2,
                   l2_penalty=1e-8,
                   **unused_params):
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    num_model=num_model or FLAGS.num_model

    


    ensemble_weights = tf.get_variable("ensemble_weights",
        [num_model],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(1)))
      
      
    ensemble_distribution = tf.nn.softmax(ensemble_weights)  # (Batch * #Labels) x (num_model )
    ensemble_distribution1=tf.cast(ensemble_distribution[0],tf.float32)
    ensemble_distribution2=tf.cast(ensemble_distribution[1],tf.float32)

    
    
    
    model1_input=tf.cast(model1_input,tf.float32)
    model2_input=tf.cast(model2_input,tf.float32)
    prediction_model_weight1=tf.multiply(model1_input,ensemble_distribution1) 
    prediction_model_weight2=tf.multiply(model2_input,ensemble_distribution2)    
    prediction_1=tf.add(prediction_model_weight1,prediction_model_weight2)

    return {"predictions": prediction_1}

class DNN(models.BaseModel):





  def create_model(self,
                   model_input,
                   **unused_params):

    reshaped_input=tf.layers.flatten(model_input)
    activations= tf.layers.dense(inputs=reshaped_input, units=1024, activation=tf.nn.relu)
    activations=tf.nn.dropout(activations, 0.8)
    activations= tf.layers.dense(inputs=activations, units=128, activation=tf.nn.relu)
    activations=tf.nn.dropout(activations, 0.8)
    activations= tf.layers.dense(inputs=activations, units=10, activation=tf.nn.softmax)

    return {"predictions": activations}

