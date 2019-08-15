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
    task = tf.add(data,1)
    config = tf.ConfigProto(device_count={'GPU': 1})
    sec = 9999
    with tf.Session(config=config) as sess:
        for _ in range(10):
            t0 = time.perf_counter()
            sess.run(task)
            t1 = time.perf_counter()
            sec = min(sec,t1-t0)
    GB = (sz*data.itemsize) / 1024 / 1024 / 1024
    print(sz*data.itemsize//1024, "  dt =", sec, " ", GB/sec, "GB/s")
    tf.reset_default_graph()