import tensorflow as tf
import numpy as np
import time

from ppf_net import inference, loss
from train_data.train import read_decode
FLAGS = tf.app.flags.FLAGS


def inputs(batch_size):
    filenames = ['./salinas_dataset/%dtest.tfrecords' % i for i in range(1, 29)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    example_list = [read_decode(filename_queue) for _ in range(8)]
    # image, label = read_decode(filename_queue)
    min_after_dequeue = FLAGS.min_after_dequeue
    capacity = min_after_dequeue + 3*batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def eval():
    with tf.Graph().as_default() as g:
        images, labels = inputs(FLAGS.batchsize)

        with tf.variable_scope('inference') as scope:
            logits = inference(images)
        predictions = tf.round(tf.sigmoid(logits))
        correct_pre = tf.equal(predictions, tf.to_float(labels))
        prob = tf.reduce_sum(tf.cast(correct_pre, tf.float32))

        variable_ave = tf.train.ExponentialMovingAverage(FLAGS.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_ave.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        sess = tf.Session()

        ckpt = tf.train.get_checkpoint_state('checkpoint/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found!')
            return

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            true_count = 0.0
            test_samples = 0
            while step < 500 and not coord.should_stop():
                print (step)
                ps = sess.run(prob)
                true_count += ps
                test_samples += FLAGS.batchsize
                step += 1
            precision = true_count / test_samples
            print ('precision: %.3f' % precision)
        # except tf.errors.OutOfRangeError:
        #     print('Done training for %d epochs, %d steps.' % (1, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    eval()