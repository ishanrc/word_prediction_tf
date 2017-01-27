import tensorflow as tf
from data_prep import get_batch

def build_model(training_data, test_data, embedding_size=50, hidden_size=200):
  vocab_size = len(training_data.labels[0])
  x = tf.placeholder(tf.float32, shape=[None, vocab_size * 3])
  y_ = tf.placeholder(tf.float32, shape=[None, vocab_size])
  
  word_to_embedding_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]))
  
  embedding_to_hidden_weights = tf.Variable(tf.truncated_normal([embedding_size * 3, hidden_size]))
  hidden_bias = tf.Variable(tf.constant(0.1, shape=[hidden_size]))

  hidden_to_output_weights = tf.Variable(tf.truncated_normal([hidden_size, vocab_size]))
  output_bias = tf.Variable(tf.constant(0.1, shape=[vocab_size]))
  
  input_words = tf.split(1, 3, x)
  word_embeddings = [tf.matmul(word, word_to_embedding_weights) for word in input_words]
  embedding_value = tf.concat(1, word_embeddings)
  hidden_value = tf.sigmoid(tf.matmul(embedding_value, embedding_to_hidden_weights) + hidden_bias)
  y = tf.matmul(hidden_value, hidden_to_output_weights) + output_bias

  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  for i in range(200000):
    batch_x, batch_y = get_batch(training_data, i, 100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={ x: test_data.inputs, y_: test_data.labels }))
