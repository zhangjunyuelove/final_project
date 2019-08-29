import tensorflow as tf
import cifar10
from datetime import datetime
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train_try',
                           '''Directory where to write event logs and checkpoint''')
tf.app.flags.DEFINE_integer('max_steps', 110000, '''Number of batches to run''')
tf.app.flags.DEFINE_boolean('log_device_placement', False, '''Whether to log device placement''')
tf.app.flags.DEFINE_integer('log_frequency', 10, '''How ofter to log results to the console''')

def stats_graph(graph):
    run_meta = tf.RunMetadata()
    flops = tf.profiler.profile(graph,run_meta=run_meta, cmd='op', options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph,run_meta=run_meta, cmd='op', options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

g=tf.Graph()
def train():
    with g.as_default() :
        
        global_step = tf.train.get_or_create_global_step()
        
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        logits = cifar10.inference3model(images)
        stats_graph(g)
        loss = cifar10.loss(logits, labels)
        train_op = cifar10.train(loss, global_step)
        
        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):

                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):

                self._step += 1
                # 在这里返回你想在运行过程中产看的信息，以list的形式传递,如:[loss, accuracy]
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):

                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    # results返回的是上面before_run()的返回结果，上面是loss所以返回loss
                    # 如若上面返回的是个list,则返回的也是个list
                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                          % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                                                      tf.train.NanTensorHook(loss),
                                                      _LoggerHook()],
                                               save_checkpoint_secs=60,
                                               config=tf.ConfigProto(
                                                   log_device_placement=FLAGS.log_device_placement
                                               )) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
