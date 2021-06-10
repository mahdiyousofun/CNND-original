import tensorflow as tf
import numpy as np
import math


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('deepth', 224,
    """deepth""")
tf.app.flags.DEFINE_integer('min_after_dequeue', 500000,
    """min_after_dequeue""")
tf.app.flags.DEFINE_integer('epochs', 1,
    """iteration epochs""")
tf.app.flags.DEFINE_integer('batchsize', 256,
	"""Number of images to process in a batch""")
tf.app.flags.DEFINE_integer('NUM_STEPS_PER_DECAY', 200000,
	"""Number of epochs to decay the learning rate""")
tf.app.flags.DEFINE_float('decay_factor', 0.1,
    """learning rate decay speed""")
tf.app.flags.DEFINE_float('initial_lr', 1e-2,
    """initial learning rate""")
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
    """average the weights""")

classes = 1


def loss(logpros, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logpros, labels=tf.to_float(labels), name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd=0.0005):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_relu(input, kernel, stride, padding):
    weights = _variable_with_weight_decay('weights', shape=kernel, stddev=math.sqrt(2.0/kernel[0]/kernel[1]/kernel[2]))
    biases = _variable_on_cpu('biases', [kernel[3]], tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, stride, padding=padding)
    return tf.nn.relu(conv+biases)


def inference(images):
    # 1x224
    with tf.variable_scope('conv1') as scope:
        conv1 = conv_relu(images, [1,3,1,20], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv2') as scope:
        conv2 = conv_relu(conv1, [1,3,20,20], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv3') as scope:
        conv3 = conv_relu(conv2, [1,3,20,20], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv5') as scope:
        conv5 = conv_relu(conv3, [1,3,20,30], [1,1,2,1], 'SAME')
    # 1x112
    with tf.variable_scope('conv6') as scope:
        conv6 = conv_relu(conv5, [1,3,30,30], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv7') as scope:
        conv7 = conv_relu(conv6, [1,3,30,30], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv8') as scope:
        conv8 = conv_relu(conv7, [1,3,30,40], [1,1,2,1], 'SAME')
    # 1x56
    with tf.variable_scope('conv9') as scope:
        conv9 = conv_relu(conv8, [1,3,40,40], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv10') as scope:
        conv10 = conv_relu(conv9, [1,3,40,40], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv11') as scope:
        conv11 = conv_relu(conv10, [1,3,40,30], [1,1,2,1], 'SAME')
    # 1x28
    with tf.variable_scope('conv12') as scope:
        conv12 = conv_relu(conv11, [1,3,30,30], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv13') as scope:
        conv13 = conv_relu(conv12, [1,3,30,30], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv14') as scope:
        conv14 = conv_relu(conv13, [1,3,30,20], [1,1,2,1], 'SAME')
    # 1x14
    with tf.variable_scope('conv15') as scope:
        conv15 = conv_relu(conv14, [1,3,20,20], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv16') as scope:
        conv16 = conv_relu(conv15, [1,3,20,20], [1,1,1,1], 'SAME')
    with tf.variable_scope('conv17') as scope:
        conv17 = conv_relu(conv16, [1,3,20,20], [1,1,2,1], 'SAME')
    # 1x7
    dims = 6
    avg_pool = tf.nn.avg_pool(conv17, [1,1,dims,1], [1,1,1,1], padding='VALID')
    with tf.variable_scope('scores') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,1,20,classes], stddev=math.sqrt(2.0/20))
        scores = tf.nn.conv2d(avg_pool, weights, [1,1,2,1], padding='VALID')
    logits_flat = tf.reshape(scores, [-1])

    return logits_flat

def trainop(total_loss, global_step):
    lr = tf.train.exponential_decay(FLAGS.initial_lr,
                                  global_step,
                                  FLAGS.NUM_STEPS_PER_DECAY,
                                  FLAGS.decay_factor,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        # opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
      FLAGS.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
