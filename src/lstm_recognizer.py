import tensorflow as tf
from tensorflow.contrib import rnn
import data_handler_lstm as data

hm_epochs = 100
n_classes = 6
batch_size = 28
chunk_size = 561
n_chunks = 47
rnn_size = 128
max_len = 47

inputs, labels = data.prepare_data(data.content, data.labels, max_len)
test_inputs, test_labels = data.prepare_data(data.test_content,
                                             data.test_labels, max_len)
dataset = tf.data.Dataset.from_tensor_slices(inputs)
dataset = dataset.batch(batch_size)
inputs = dataset

x = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
y = tf.placeholder(tf.float32)


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,
                                                       labels=y)
            )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(len(inputs)/batch_size)):
                epoch_x = inputs[i]
                epoch_y = labels[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',
                  hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_inputs.reshape(
                    (-1, n_chunks, chunk_size)), y: test_labels}))


train_neural_network(x)
