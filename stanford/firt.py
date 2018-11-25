# tensorboard --logdir="./graphs" --port 6006

import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
	# writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
	print(sess.run(x))
writer.close()
