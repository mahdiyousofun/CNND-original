import tensorflow as tf
import numpy as np
import time

from ppf_net import inference, loss, trainop

FLAGS = tf.app.flags.FLAGS


def read_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([1*FLAGS.deepth])
    image = tf.reshape(image, [1, FLAGS.deepth, 1])
    image = tf.cast(image, tf.float32) * (1. / 255)
    # preprocess
    image = tf.image.per_image_standardization(image)
    #image = tf.slice(image, [0, 0, 0], [1, 224, 1])  # slice
    label = tf.cast(features['label'], tf.int32)
    return image, label


def inputs(batch_size, num_epochs):
    filenames = ['salinas_dataset/%dtrain.tfrecords' % i for i in range(1, 29)]
    for f in filenames:
        if not tf.gfile.Exists(f):
           raise ValueError('Failed to find file: ' + f)
    # string_input_producer has options for shuffling and setting a maximum number of epochs
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_decode(filename_queue) for _ in range(8)]
    # image, label = read_decode(filename_queue)
    min_after_dequeue = FLAGS.min_after_dequeue
    capacity = min_after_dequeue + 3*batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def train(reuse=None):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images, labels = inputs(FLAGS.batchsize, FLAGS.epochs)
        with tf.variable_scope('inference', reuse=reuse) as scope:
            logits = inference(images)
        loss_ = loss(logits, labels)
        # tf.scalar_summary('loss', loss_)
        # train_op = tf.train.AdamOptimizer(1e-3).minimize(loss_)
        train_op = trainop(loss_, global_step)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter('sl')
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss_])
                duration = time.time() - start_time

                if step % 100 == 0:
                    print('Step %d: loss = %.5f (%.3f sec)' % (step, loss_value, duration))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                if step % 1000 == 0:
                    saver.save(sess, 'checkpoint/model.ckpt', global_step=step)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
     train(None)
