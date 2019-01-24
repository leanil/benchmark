import numpy as np
import tensorflow as tf
import time

def init(*args):
    if args[0] == None:
        return np.random.rand(*args[1:])
    #tf.random_uniform(args[1:])
    else:
        return args[0]
        
def measure_transfer(*shape):
    t0 = time.perf_counter()
    data = init(None,*shape)
    t1 = time.perf_counter()
    with tf.Session() as sess:
        res = sess.run(tf.add(data,1))
    t2 = time.perf_counter()
    return (t1-t0,t2-t1)
    