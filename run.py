"""
run.py

Core script for building, training, and evaluating a Recurrent Entity Network.
"""
from model.entity_network import EntityNetwork
from preprocessor.reader import parse
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("task_id", 1, "ID of Task to Train/Evaluate [1 - 20].")
tf.app.flags.DEFINE_string("data_path", "tasks/en-valid-10k", "Path to Training Data")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of word embeddings.")
tf.app.flags.DEFINE_integer("memory_slots", 20, "Number of dynamic memory slots.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("num_epochs", 200, "Number of Training Epochs.")
tf.app.flags.DEFINE_float("learning_rate", .01, "Learning rate for ADAM Optimizer.")
tf.app.flags.DEFINE_integer("decay_epochs", 25, "Number of epochs to run before learning rate decay.")
tf.app.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float('clip_gradients', 40.0, 'Norm to clip gradients to.')


def main(_):
    # Get Vectorized Forms of Stories, Questions, and Answers
    trainS, trainS_len, trainQ, trainA, word2id = parse(FLAGS.data_path, FLAGS.task_id, "train")
    valS, valS_len, valQ, valA, _ = parse(FLAGS.data_path, FLAGS.task_id, "valid", word2id=word2id)
    testS, testS_len, testQ, testA, _ = parse(FLAGS.data_path, FLAGS.task_id, "test", word2id=word2id)

    # Assert Shapes
    assert(trainS.shape[1:] == valS.shape[1:] == testS.shape[1:])
    assert(trainQ.shape[1] == valQ.shape[1] == testQ.shape[1])

    # Build Model
    with tf.Session() as sess:
        # Instantiate Model
        entity_net = EntityNetwork(word2id, trainS.shape[2], trainS.shape[1], FLAGS.batch_size,
                                   FLAGS.memory_slots, FLAGS.embedding_size, FLAGS.learning_rate, 
                                   FLAGS.decay_epochs * (trainS.shape[0]/FLAGS.batch_size), FLAGS.decay_rate)
        
        # Initialize all Variables
        sess.run(tf.global_variables_initializer())

        # Start Training Loop
        n, bsz = trainS.shape[0], FLAGS.batch_size
        for epoch in range(FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, n, bsz), range(bsz, n, bsz)):
                curr_loss, curr_acc, logits, _ = sess.run([entity_net.loss_val, entity_net.accuracy, entity_net.logits, entity_net.train_op], 
                                                  feed_dict={entity_net.S: trainS[start:end], 
                                                             entity_net.S_len: trainS_len[start:end],
                                                             entity_net.Q: trainQ[start:end],
                                                             entity_net.A: trainA[start:end]})
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 100 == 0:
                    print "Epoch %d Batch %d Average Loss:" % (epoch, counter), loss / float(counter), "Average Accuracy:", acc / float(counter)

if __name__ == "__main__":
    tf.app.run()