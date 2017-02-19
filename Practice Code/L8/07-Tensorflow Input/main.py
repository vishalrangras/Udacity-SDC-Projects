import tensorflow as tf

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)
with tf.Session() as sess:
    output = sess.run(x,feed_dict={x:'Hello World'})
    print(output)
    output = sess.run(x,feed_dict={x:'Test String', y:123, z:45.67})
    print(output)

def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x,feed_dict={x:123})

    return output

print(run())
