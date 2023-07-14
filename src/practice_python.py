import tensorflow as tf


x = tf.constant(11.1, shape=(1,1));
x = tf.constant([[1,2,3],[4,5,6]]);
x = tf.ones((3,3));
x = tf.eye(3)

x = tf.random.normal((3,3),mean=0,stddev=1)
x = tf.random.uniform((4,4),minval=-2,maxval=2.2)

x = tf.range(0.3,0.8,0.13)
x = tf.random.uniform((3,3),-8,0)
delta = 0.14;
x = tf.range(0.3,1.9,delta);
x = tf.cast(x, dtype=tf.float32)

y = tf.cast(x, dtype=tf.float32)

z = tf.tensordot(x, y, axes=1) # cannot  do it if type is different

print(x)

print(x[0:1])

x_gathered = tf.gather(x, (1,3,4))
print(x_gathered)

words = ['godzilla', 'darkness', 'leaving heaven'];
