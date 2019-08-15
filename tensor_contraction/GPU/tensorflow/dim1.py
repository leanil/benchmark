from util import config, init
import tensorflow as tf
from tensorflow.python.client import timeline

i = 10000000

def dot(A=None, B=None):
    A = init(A,i)
    B = init(B,i)
    task = tf.tensordot(A,B,1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def dot1(A=None, B=None, C=None):
    A = init(A,i)
    B = init(B,i)
    C = init(C,i)
    task = tf.reduce_sum(tf.multiply(tf.multiply(A,B),C))
    with tf.Session(config = config) as sess:
        return sess.run(task)

def dot2(A=None, B=None, C=None):
    A = init(A,i)
    B = init(B,i)
    C = init(C,i)
    task = tf.tensordot(tf.add(A,B),C,1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def dot3(A=None, B=None, C=None):
    A = init(A,i)
    B = init(B,i)
    C = init(C,i)
    task = tf.tensordot(tf.add(A,B),tf.subtract(A,C),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def dot4(A=None, B=None, C=None, D=None):
    A = init(A,i)
    B = init(B,i)
    C = init(C,i)
    D = init(D,i)
    task = tf.tensordot(tf.add(A,B),tf.subtract(C,D),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def dot5(a=None, A=None, b=None, B=None, c=None, C=None, d=None, D=None):
    a = init(a)
    A = init(A,i)
    b = init(b)
    B = init(B,i)
    c = init(c)
    C = init(C,i)
    d = init(d)
    D = init(D,i)
    task = tf.tensordot(tf.add(tf.multiply(A,a),tf.multiply(B,b)),tf.add(tf.multiply(C,c),tf.multiply(D,d)),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

# Can't force temorary
def dot6(A=None, B=None, C=None, D=None):
    A = init(A,i)
    B = init(B,i)
    C = init(C,i)
    D = init(D,i)
    task = tf.tensordot(tf.add(A,B),tf.subtract(C,D),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)
