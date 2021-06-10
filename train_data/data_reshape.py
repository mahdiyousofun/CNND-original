from __future__ import division

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

data = sio.loadmat('Salinas.mat') # 512*217*204
gt = sio.loadmat('Salinas_gt.mat')
X = data['salinas']
plt.figure(3),plt.imshow(X[:,:,223])
# normalize X to 0-255
X -= np.amin(X)
X = X / np.amax(X)
X *= 255
X = np.int16(X)
print (X[10, 100, :])
Y = gt['salinas_gt']
plt.figure(1),plt.imshow(Y)
Y[Y==8] = 0
Y[Y==9] = 8
Y[Y==10] = 9
Y[Y==11] = 10
Y[Y==12] = 11
Y[Y==13] = 12
Y[Y==14] = 13
Y[Y==15] = 0
Y[Y==16] = 14
plt.figure(2), plt.imshow(Y)
plt.show()