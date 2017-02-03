import tensorflow as tf
from data_prep import get_batch
from data_prep import Model
from data_prep import LabeledData

def build(vocab_size, embedding_size=50, hidden_size=200):
  x = tf.placeholder(tf.float32, shape=[None, vocab_size * 3])
  y_ = tf.placeholder(tf.float32, shape=[None, vocab_size])
  
  word_to_embedding_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]))
  
  embedding_to_hidden_weights = tf.Variable(tf.truncated_normal([embedding_size * 3, hidden_size]))
  hidden_bias = tf.Variable(tf.constant(0.0, shape=[hidden_size]))

  hidden_to_output_weights = tf.Variable(tf.truncated_normal([hidden_size, vocab_size]))
  output_bias = tf.Variable(tf.constant(0.0, shape=[vocab_size]))
  
  input_words = tf.split(1, 3, x)
  word_embeddings = [tf.matmul(word, word_to_embedding_weights) for word in input_words]
  embedding_value = tf.concat(1, word_embeddings)
  hidden_value = tf.sigmoid(tf.matmul(embedding_value, embedding_to_hidden_weights) + hidden_bias)
  y = tf.matmul(hidden_value, hidden_to_output_weights) + output_bias

  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return Model(x, y_, y, cross_entropy, accuracy)

def evaluate(session, model, test_data):
  return session.run(model.accuracy, feed_dict={ model.x: test_data.inputs, model.y_: test_data.labels })

def train(session, 
  model,
  training_data,
  validation_data,
  test_data,
  steps=2000,
  batch_size=100,
  validation_freq=None):
  train_step = tf.train.AdamOptimizer(epsilon=1e-4).minimize(model.loss)
  session.run(tf.global_variables_initializer())
  for i in range(steps):
    batch_x, batch_y = get_batch(training_data, i, batch_size)
    session.run(train_step, feed_dict={model.x: batch_x, model.y_: batch_y})
    if validation_freq != None and i % validation_freq == 0:
      print('step %d: validation accuracy: %f train accuracy: %f' % (i, evaluate(session, model, validation_data), evaluate(session, model, LabeledData(batch_x, batch_y))))

  print('final: validation accuracy: %f test accuracy: %f' % (evaluate(session, model, validation_data), evaluate(session, model, test_data)))
