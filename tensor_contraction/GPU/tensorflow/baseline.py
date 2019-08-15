from util import config, init
import tensorflow as tf

i = 10000000

def baseline_sum(A_host=None):
        A = init(A_host,i)
        task = tf.reduce_sum(A)
        with tf.Session(config = config) as sess:
            return sess.run(task)

def baseline_inc(A=None):
    A = init(A,i)
    task = tf.add(A,1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def baseline_prod(A=None, c = None):
        A = init(A,i)
        c = init(c)
        task = tf.multiply(A,c)
        with tf.Session(config = config) as sess:
            return sess.run(task)

#i,j,k = 2000,2000,2000

#def mat_add(i,j):
#    t0 = time.perf_counter()
#    A = np.random.rand(i,j)
#    B = np.random.rand(i,j)
#    task = tf.add(A,B)
#    with tf.Session() as sess:
#        e = sess.run(task)
#    t1 = time.perf_counter()
#    print(t1 - t0)

#def t10():
#    # for i in range(500,2000,100):
#        # j,k = i,i
#        e = np.array(i)
#        A = np.random.rand(i,k)
#        B = np.random.rand(k,j)
#        C = np.random.rand(j,k)
#        D = np.random.rand(k)
#        con_dims = [[1],[0]]
#        mm = tf.tensordot(A,B,con_dims,name="MatMat")
#        mv = tf.tensordot(C,D, con_dims,name="MatVec1")
#        task = tf.tensordot(mm,mv, con_dims,name="MatVec2")
#        with tf.Session() as sess:
#            for _ in range(10):
#                t0 = time.perf_counter()
#                e = sess.run(task)
#                t1 = time.perf_counter()
#                print(t1 - t0)
