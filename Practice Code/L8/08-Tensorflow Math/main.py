import tensorflow as tf

x = tf.add(5,2)
y = tf.subtract(10,4)
z = tf.multiply(2,5)
w = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))

with tf.Session() as sess:
    output = sess.run(x)
    print(output)
    output = sess.run(y)
    print(output)
    output = sess.run(z)
    print(output)
    output = sess.run(w)
    print(output)

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1
x = tf.constant(x)
y = tf.constant(y)

# TODO: Print z from a session
z = tf.subtract(tf.cast(tf.divide(x,y),tf.int32),tf.constant(1))
print()
print("------------Algorithm Solution starts here------------")
print()
with tf.Session() as sess:
    print("x = ",sess.run(x))
    print("y = ",sess.run(y))
    print("z = x/y - 1, which evaluates to: ",sess.run(z))
