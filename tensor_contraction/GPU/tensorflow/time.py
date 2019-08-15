import tensorflow as tf
from tensorflow.python.client import timeline

print(tf.test.is_gpu_available())
n = 1000
x = tf.random_normal([n, n])
y = tf.random_normal([n, n])
res = tf.matmul(x, y)

# Run the graph with full trace option
with tf.Session() as sess:
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()
    print(sess.run(res))#, options=run_options, run_metadata=run_metadata)

    # Create the Timeline object, search MatMul node, get relative time
    #tl = timeline.Timeline(run_metadata.step_stats)
    #for ds in run_metadata.step_stats.dev_stats:
    #    for ns in ds.node_stats:
    #        if ns.node_name == "MatMul:MatMul":
    #            print(n, ns.all_end_rel_micros / 1000.0 / 1000.0, "seconds")
#Reset the node names used (so that in the next iteration we still find MatMul and not MatMul_2)
#tf.reset_default_graph()