from util import config, init
import tensorflow as tf
from tensorflow.python.client import timeline

i, j, k = 12000, 12001, 12002

def t1(A=None, B=None):
    A = init(A,i,j)
    B = init(B,j)
    task = tf.tensordot(A,B,1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t2(A=None, B=None, C=None):
    A = init(A,i,j)
    B = init(B,j)
    C = init(C,i)
    task = tf.multiply(tf.tensordot(A,B,1),C)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t3(A=None, B=None, C=None):
    A = init(A,i,j)
    B = init(B,i,j)
    C = init(C,j)
    task = tf.tensordot(tf.add(A,B),C,1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t4(A=None, B=None, C=None, D=None):
    A = init(A,i,j)
    B = init(B,i,j)
    C = init(C,j)
    D = init(D,j)
    task = tf.tensordot(tf.add(A,B),tf.add(C,D),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t5(a=None, A=None, b=None, B=None, c=None, C=None, d=None, D=None):
    a = init(a)
    A = init(A,i,j)
    b = init(b)
    B = init(B,i,j)
    c = init(c)
    C = init(C,j)
    d = init(d)
    D = init(D,j)
    task = tf.tensordot(tf.add(tf.multiply(a,A),tf.multiply(b,B)),tf.add(tf.multiply(c,C),tf.multiply(d,D)),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t6(A=None, B=None, C=None, D=None):
    A = init(A,i,1)
    B = init(B,j)
    C = init(C,i,1)
    D = init(D,j)
    task = tf.reduce_sum(tf.multiply(tf.multiply(A,B),tf.multiply(C,D)),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t7(A=None, C=None, D=None):
    A = init(A,i,j)
    C = init(C,j,k)
    D = init(D,k)
    task = tf.tensordot(A,tf.tensordot(C,D,1),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t8(A=None, B=None, C=None, D=None):
    A = init(A,i,j)
    B = init(B,i,j)
    C = init(C,j,k)
    D = init(D,k)
    task = tf.tensordot(tf.add(A,B),tf.tensordot(C,D,1),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t9(A=None, B=None, C=None, D=None):
    A = init(A,i,k)
    B = init(B,k,j)
    C = init(C,j)
    D = init(D,j)
    task = tf.tensordot(tf.tensordot(A,B,1),tf.add(C,D),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)

def t10(A=None, B=None, C=None, D=None):
    A = init(A,i,k)
    B = init(B,k,j)
    C = init(C,j,k)
    D = init(D,k)
    task = tf.tensordot(tf.tensordot(A,B,1),tf.tensordot(C,D,1),1)
    with tf.Session(config = config) as sess:
        return sess.run(task)