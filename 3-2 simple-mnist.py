
# coding: utf-8

# In[8]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# In[9]:


# load data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size
print (n_batch)


# In[15]:


# model
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#w = tf.Variable(tf.random_normal([784, 10]))
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(x, w) + b)


loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:
# https://blog.csdn.net/wangkun1340378/article/details/72782593
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        if epoch%10 == 0:
            print ("iter: " + str(epoch) + "test acc: " + str(acc))

