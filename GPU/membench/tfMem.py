import numpy as np
import tensorflow as tf
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
szmin = 2
szmax = 28
for i in range(szmin,szmax+1):
    sz = 2 ** (i-1)
    data = np.random.rand(sz)
    t0 = time.perf_counter()
    with tf.Session() as sess:
        sess.run(tf.add(data,1))
    t1 = time.perf_counter()
    sec = t1-t0
    GB = (sz*data.itemsize) / 1024 / 1024 / 1024
    print(sz*data.itemsize//1024, "  dt =", sec, " ", GB/sec, "GB/s")