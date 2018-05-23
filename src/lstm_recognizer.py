import tensorflow as tf



batch_size = 561
lstm_size = 561
time_steps = 7352

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
hidden_state = tf.zeros([batch_size, lstm.state_size])
current_state = tf.zeros([batch_size, lstm.state_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0

