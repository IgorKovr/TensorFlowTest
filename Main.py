# The canonical import statement for TensorFlow programs is as follows:
import tensorflow as tf

# A computational graph is a series of TensorFlow operations arranged into a graph of nodes.

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly

# A session encapsulates the control and state of the TensorFlow runtime.

sess = tf.Session()

# A placeholder is a promise to provide a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# + provides a shortcut for tf.add(a, b)

adder_node = a + b

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Variables allow us to add trainable parameters to a graph.
# They are constructed with a type and initial value:

w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w * x + b

# To initialize all the variables in a TensorFlow program,
# you must explicitly call a special operation as follows:

init = tf.global_variables_initializer()

# Until we call sess.run, the variables are uninitialized.

sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# A loss function measures how far apart the current model is from the provided data.

y = tf.placeholder(tf.float32)

# We call tf.square to square that error

squared_deltas = tf.square(linear_model - y)

# Then, we sum all the squared errors to create a single scalar
# that abstracts the error of all examples using tf.reduce_sum:

loss = tf.reduce_sum(squared_deltas)

# producing the loss value:

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# A variable is initialized to the value provided to tf.Variable
# but can be changed using operations like tf.assign

fixW = tf.assign(w, [-1.])
fixB = tf.assign(b, [1.])

sess.run([fixW, fixB])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#
# tf.train API
#

# Optimizers slowly change each variable in order to minimize the loss function
# GradientDescentOptimizer modifies each variable according to the magnitude of the derivative of loss
# with respect to that variable

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
sess.run(init)  # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([w, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
