import os
from util import init, measure_transfer
import tensorflow as tf
import numpy as np
import time

i = 10000000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def baseline_sum(repeat, A_host=None, a_host=None):
        A = init(A_host,i)
        task = tf.reduce_sum(A)
        times = []
        #for i in range(repeat):
        with tf.Session() as sess:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(task, options=run_options, run_metadata=run_metadata)

            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            for ds in run_metadata.step_stats.dev_stats:
                print(ds)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)
#i,j,k = 1000,1000,1000

def mat_add(i,j):
    t0 = time.perf_counter()
    A = np.random.rand(i,j)
    B = np.random.rand(i,j)
    task = tf.add(A,B)
    with tf.Session() as sess:
        e = sess.run(task)
    t1 = time.perf_counter()
    print(t1-t0)

def t10():
    for i in range(500,2000,100):
        j,k = i,i
        t0 = time.perf_counter()
        e = np.array(i)
        A = np.random.rand(i,k)
        B = np.random.rand(k,j)
        C = np.random.rand(j,k)
        D = np.random.rand(k)
        con_dims = [[1],[0]]
        mm = tf.tensordot(A,B,con_dims,name="MatMat")
        mv = tf.tensordot(C,D, con_dims,name="MatVec1")
        task = tf.tensordot(mm,mv, con_dims,name="MatVec2")
        with tf.Session() as sess:
                e = sess.run(task)
        t1 = time.perf_counter()
        print(t1-t0)

#baseline_sum(1)
for i in range(1,33):
    t = measure_transfer(i*1000000)
    print(t[0],t[1],t[0]+t[1])