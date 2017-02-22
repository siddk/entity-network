"""
run.py

Core script for building, training, and evaluating a Recurrent Entity Network.
"""
from preprocessor.reader import parse
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("task_id", 1, "ID of Task to Train/Evaluate [1 - 20].")
tf.app.flags.DEFINE_string("data_path", "tasks/en-valid-10k", "Path to Training Data")

def main(_):
    # Get Vectorized Forms of Stories, Questions, and Answers
    trainS, trainS_len, trainQ, trainA, word2id = parse(FLAGS.data_path, FLAGS.task_id, "train")
    valS, valS_len, valQ, valA, _ = parse(FLAGS.data_path, FLAGS.task_id, "valid", word2id=word2id)
    testS, testS_len, testQ, testA, _ = parse(FLAGS.data_path, FLAGS.task_id, "test", word2id=word2id)

    # Assert Shapes
    assert(trainS.shape[1:] == valS.shape[1:] == testS.shape[1:])
    assert(trainQ.shape[1] == valQ.shape[1] == testQ.shape[1])

    # Build Model
    entity_net = EntityNetwork(word2id)

if __name__ == "__main__":
    tf.app.run()