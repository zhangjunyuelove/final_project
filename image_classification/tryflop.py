import tensorflow as tf
import os
import re
import input
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow.contrib.slim as slim

# tf.app.flags主要处理命令行参数的解析工作
FLAGS = tf.app.flags.FLAGS

# tf.app.flag.DEFINE_xxx()就是添加命令行的可选参数
tf.app.flags.DEFINE_integer('batch_size', 128, '''Number of images to process in  a batch''')
tf.app.flags.DEFINE_string('data_dir', 'E:/Moe_Junyue',
                           '''Path to the CIFAR-10 data directory''')
tf.app.flags.DEFINE_boolean('use_fp16', False, '''Train the model using fp16''')

# 参数设置
IMAGE_SIZE = input.IMAGE_SIZE
NUM_CLASS = input.NUM_CLASSES
NUM_EXAMPLE_PER_EPOCH_FOP_TRAIN = input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLE_PER_EPOCH_FOR_TEST = input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST


MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
# 衰减系数decay_rate
LEARNING_RATE_DECAY_FACTOR = 0.1
# 初始学习率
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# 为了在多个GPU上共享变量，所有的变量都绑定在CPU上，并通过tf.get_variable()访问
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


# weight decay 是放在正则项前面的一个系数，控制正则化在损失函数中所占的权重，正则项一般表示模型的复杂度，
# 所以weight decay的作用
# 是调节模型复杂度对损失函数的影响
def _variable_with_weight_decay(name, shape, stddev, wd):
    '''用weight decay 建立一个初始的变量
    @param name:
    @param shape:
    @param stddev: 截断高斯分布的标准偏差
    @param wd: 如果wd不为None, 为变量添加L2_loss并与权重衰减系数相乘
    @return: 张量
    '''
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float16
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        # 添加L2Loss, 并将其添加到‘losses’集合
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
with tf.Graph().as_default() as graph:
    A = tf.get_variable(initializer=tf.random_normal_initializer(dtype=tf.float32), shape=(128, 24,24,3), name='A')

    images=tf.layers.flatten(A)

    images_reshape=slim.fully_connected(
         images, 512, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(1e-8))
    
    
    images_reshape=slim.fully_connected(
         images_reshape, 120, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(1e-8))
    
    
    
    logits=slim.fully_connected(
         images_reshape, NUM_CLASS, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(1e-8))


    
    stats_graph(graph)
