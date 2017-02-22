"""
entity_network.py

Model definition class for the Recurrent Entity Network. Defines Input Encoder,
Dynamic Memory Cell, and Readout components.
"""
import tensorflow as tf

class EntityNetwork():
    def __init__(self, vocabulary, sentence_len, story_len):
        """
        Initialize an Entity Network with the necessary hyperparameters.

        :param vocabulary: Word Vocabulary for given model.
        :param sentence_len: Maximum length of a sentence.
        :param story_len: Maximum length of a story.
        """
        self.vocabulary, self.sentence_len, self.story_len = vocabulary, max_sentence, story_len

        # Setup Placeholders
        self.S = tf.placeholder(tf.int32, [None, self.story_len, self.sentence_len], name="Story")
        self.S_len = tf.placeholder(tf.int32, [None], name="Story_Length")
        self.Q = tf.placeholder(tf.int32, [None, self.sentence_len], name="Query")
        self.A = tf.placeholder(tf.int32, [None], name="Answer")

        # Instantiate Network Weights

