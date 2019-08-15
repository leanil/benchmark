import tensorflow as tf
from tensorflow.python.client import timeline
import timeit

X = tf.random_uniform([1 << 20])
sum = tf.reduce_sum(X)
# Create a session object
with tf.Session() as sess:
  #result = tensorflow.matmul(A, B)
  timer = timeit.Timer("sess.run(sum)", setup="import tensorflow as tf; from __main__ import sess, X, sum")
  tensorflow_times_list = timer.repeat(10, 1)
  print(min(tensorflow_times_list)*1000,"ms")
  print(tensorflow_times_list)

with tf.Session() as sess:
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  sess.run(sum, options=options, run_metadata=run_metadata)
  print(run_metadata.step_stats)

# Create the Timeline object, and write it to a json file
#fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#chrome_trace = fetched_timeline.generate_chrome_trace_format()
#with open('timeline_01.json', 'w') as f:
#    f.write(chrome_trace)