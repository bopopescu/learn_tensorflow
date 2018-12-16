import tensorflow as tf
from utils import read_data_from_file

file_name = './birth_life_2010.txt'
data, n_samples = read_data_from_file(file_name)

X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32, name = 'Y')

w = tf.get_variable('weight', initializer = tf.constant(0.0))
b = tf.get_variable('bias', initializer = tf.constant(0.0))

Y_predict = w * X + b

loss = tf.square(Y_predict - Y, name = 'loss')

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)

writer = tf.summary.FileWriter('./graph', tf.get_default_graph())

iteration = 100
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for _ in range(iteration):
		for x, y in data:
			sess.run(optimizer, feed_dict={X:x, Y:y}) 

	w_out, b_out = sess.run([w, b])
	
writer.close()
print(w_out, b_out)

