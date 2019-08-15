import csv
import numpy as np
import subprocess
import tensorflow as tf
import time

# restrict to 1 GPU
config = tf.ConfigProto(device_count={'GPU': 1})#, log_device_placement=True)

def init(*args):
    if args[0] == None:
        if  len(args) == 1:
            return np.float32(np.random.rand())
        return np.random.rand(*args[1:]).astype("float32")
    #tf.random_uniform(args[1:])
    else:
        return args[0]
        
def nvprof(i):
    nvprof_args = ["nvprof", "--unified-memory-profiling", "off", "--print-gpu-summary", "--csv", "python3", "tensorflow_test.py"]
    time_units = {"s":1, "ms":1e-3, "us":1e-6}
    out = subprocess.check_output(nvprof_args + [str(i)], stderr=subprocess.STDOUT, encoding="utf-8")
    #print(out)
    parser = csv.reader(out.splitlines())
    calc, copy = 0,0
    for row in parser:
        if row[0] == "GPU activities":
            if row[7].find("memcpy") != -1:
                copy += float(row[2])
            else:
                calc += float(row[2])
        elif len(row) > 2 and row[2] in time_units:
           unit = time_units[row[2]]
    return(copy * unit, calc * unit)

def measure_transfer(*shape):
    data = init(None,*shape)
    min_time = 999999
    with tf.Session() as sess:
        for i in range(10):
            t0 = time.perf_counter()
            sess.run(tf.add(data,1))
            t1 = time.perf_counter()
            min_time = min(min_time, t1 - t0)
    return min_time
    