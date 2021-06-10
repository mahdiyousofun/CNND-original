# coding: utf-8

# In[1]:

# get_ipython().magic(u'matplotlib inline')
import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from ppf_net import inference

# dataf = 'cuprite.mat'
# dataf = 'Airport.mat'
dataf = 'San_Diego.mat'
data = sio.loadmat(dataf)
# dtype = int16
label = data['map']
data = data['data']
# normalize data to 0-255

# ---------- for image beach ----
# rows = 150
# cols = 150
# deepth = 188
# ---------- for image beach ----

# ---------- for image sandiego ----
rows = 100
cols = 100
deepth = 189
# ---------- for image beach ----

# ---------- for image airport ----
# rows = 100
# cols = 100
# deepth = 191
# ---------- for image airport ----

data = (data - np.amin(data)) / float(np.amax(data))
data = np.reshape(data, (rows, cols, deepth))


# labelf = 'cuprite_mask_01.mat'
# labelf = 'PlaneGT.mat'
# label = sio.loadmat(labelf)['map']
# label = np.reshape(label, (rows, cols))


def dual_window(data, row, col, in_window=1, out_window=2):
    total = 0
    h, w, deepth = data.shape
    for i in range(-np.int32(out_window/2), np.int32(out_window/2) + 1):
        for j in range(-np.int32(out_window/2), np.int32(out_window/2) + 1):
            if -np.int32(in_window/2) <= i <= np.int32(in_window/2) and -np.int32(in_window/2) <= j <= np.int32(in_window/2):
                continue
            r = row + i
            c = col + j
            if r < 0 or r >= h:
                continue
            if c < 0 or c >= w:
                continue
            total += 1
    res = np.zeros((total, 2, deepth, 1))
    index = 0
    for i in (-np.int32(out_window/2), np.int32(out_window/2) + 1):
        for j in (-np.int32(out_window/2), np.int32(out_window/2) + 1):
            if -np.int32(in_window/2) <= i <= np.int32(in_window/2) and -np.int32(in_window/2) <= j <= np.int32(in_window/2):
                continue
            r = row + i
            c = col + j
            if r < 0 or r >= h:
                continue
            if c < 0 or c >= w:
                continue
            res[index, 0, :, 0] = data[row, col, :]
            res[index, 1, :, 0] = data[r, c, :]
            index += 1
    return res




def single_window(data, row, col, step=1):
    total = 0
    h, w, deepth = data.shape
    for i in range(-step, step + 1):
        for j in range(-step, step + 1):
            if i == 0 and j == 0:
                continue
            r = row + i
            c = col + j
            if r < 0 or r >= h:
                continue
            if c < 0 or c >= w:
                continue
            total += 1
    index = 0
    res = np.zeros((total, 2, deepth, 1))
    for i in range(-step, step + 1):
        for j in range(-step, step + 1):
            if i == 0 and j == 0:
                continue
            r = row + i
            c = col + j
            if r < 0 or r >= h:
                continue
            if c < 0 or c >= w:
                continue
            res[index, 0, :, 0] = data[row, col, :]
            res[index, 1, :, 0] = data[r, c, :]
            index += 1
    return res


images = tf.placeholder(tf.float32, shape=(None, 2, deepth, 1))
with tf.device('/cpu:0'):
    with tf.variable_scope('inference') as scope:
        logits = tf.reshape(inference(images), [-1, 1])
    prediction = tf.reduce_mean(tf.sigmoid(logits))
sess = tf.InteractiveSession()

FLAGS = tf.app.flags.FLAGS
variable_ave = tf.train.ExponentialMovingAverage(FLAGS.MOVING_AVERAGE_DECAY)
variable_to_restore = variable_ave.variables_to_restore()
saver = tf.train.Saver(variable_to_restore)
predictions = np.zeros((rows, cols))
ckpt = tf.train.get_checkpoint_state('checkpoint/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print('No checkpoint file found!')
start = time.time()
# predictions[10, 10] = prediction.eval(feed_dict={images:single_window(data, 10, 10, 2)})
for i in range(rows):
    for j in range(cols):
        predictions[i, j] = prediction.eval(feed_dict={images:single_window(data, i, j, 7)})
        # predictions[i, j] = prediction.eval(feed_dict={images: dual_window(data, i, j, 1, 7)})
print(time.time() - start)

# predictions = 1 - predictions
sio.savemat('tests/cnndRES.mat', {'data': predictions})
plt.imshow(predictions, cmap='gray')
plt.show()
predictions = np.reshape(predictions, rows * cols)
mask = np.reshape(label, rows * cols)
anomaly_map = (mask == 1)
normal_map = (mask == 0)
r_max = np.amax(predictions)
r_min = np.amin(predictions)
taus = np.linspace(r_min, r_max, 5000)
PF = np.zeros(5000)
PD = np.zeros(5000)
index = 0
for i in taus:
    anomaly_map_cnn = (predictions >= i)
    PF[index] = np.sum(anomaly_map_cnn & normal_map) / float(np.sum(normal_map))
    PD[index] = np.sum(anomaly_map_cnn & anomaly_map) / float(np.sum(anomaly_map))
    index += 1
area = (np.sum((PF[0:-1] - PF[1:]) * (PD[1:] + PD[0:-1]) / 2.0))
print(area)
sio.savemat('tests/cnndROCpf.mat', {'data': PF})
sio.savemat('tests/cnndROCpd.mat', {'data': PD})
plt.plot(PF, PD)
plt.show()
