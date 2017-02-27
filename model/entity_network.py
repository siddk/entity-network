"""
entity_network.py

Model definition class for the Recurrent Entity Network. Defines Input Encoder,
Dynamic Memory Cell, and Readout components.
"""
from collections import namedtuple
from tflearn.activations import sigmoid, softmax
import functools
import tensorflow as tf
import tflearn

PAD_ID = 0

def prelu_func(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
prelu = functools.partial(prelu_func, initializer=tf.constant_initializer(1.0))


class EntityNetwork():
    def __init__(self, vocabulary, sentence_len, story_len, batch_size, memory_slots, embedding_size, 
                 learning_rate, decay_steps, decay_rate, clip_gradients=40.0, 
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Initialize an Entity Network with the necessary hyperparameters.

        :param vocabulary: Word Vocabulary for given model.
        :param sentence_len: Maximum length of a sentence.
        :param story_len: Maximum length of a story.
        """
        self.vocab_sz, self.sentence_len, self.story_len = len(vocabulary), sentence_len, story_len
        self.embed_sz, self.memory_slots, self.init = embedding_size, memory_slots, initializer
        self.bsz, self.lr, self.decay_steps, self.decay_rate = batch_size, learning_rate, decay_steps, decay_rate
        self.clip_gradients = clip_gradients

        # Setup Placeholders
        self.S = tf.placeholder(tf.int32, [None, self.story_len, self.sentence_len], name="Story")
        self.S_len = tf.placeholder(tf.int32, [None], name="Story_Length")
        self.Q = tf.placeholder(tf.int32, [None, self.sentence_len], name="Query")
        self.A = tf.placeholder(tf.int64, [None], name="Answer")

        # Setup Global, Epoch Step 
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # Instantiate Network Weights
        self.instantiate_weights()

        # Build Inference Pipeline
        self.logits = self.inference()

        # Build Loss Computation
        self.loss_val = self.loss()

        # Build Training Operation
        self.train_op = self.train()

        # Create operations for computing the accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.A)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
    
    def instantiate_weights(self):
        """
        Instantiate Network Weights, including all weights for the Input Encoder, Dynamic
        Memory Cell, as well as Output Decoder.
        """
        # Create Embedding Matrix, with 0 Vector for PAD_ID (0)
        E = tf.get_variable("Embedding", [self.vocab_sz, self.embed_sz], initializer=self.init)
        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_sz)], 
                                    dtype=tf.float32, shape=[self.vocab_sz, 1])
        self.E = E * zero_mask
        
        # Create Learnable Mask
        self.story_mask = tf.get_variable("Story_Mask", [self.sentence_len, 1], initializer=tf.constant_initializer(1.0))
        self.query_mask = tf.get_variable("Query_Mask", [self.sentence_len, 1], initializer=tf.constant_initializer(1.0))

        # Create Memory Cell Keys [IF DESIRED - TIE KEYS HERE]
        self.keys = [tf.get_variable("Key_%d" % i, [self.embed_sz], initializer=self.init) 
                         for i in range(self.memory_slots)]
        
        # Create Memory Cell
        self.cell = DynamicMemory(self.memory_slots, self.embed_sz, self.keys)

        # Output Module Variables
        self.H = tf.get_variable("H", [self.embed_sz, self.embed_sz], initializer=self.init)
        self.R = tf.get_variable("R", [self.embed_sz, self.vocab_sz], initializer=self.init)

    def inference(self):
        """
        Build inference pipeline, going from the story and question, through the memory cells, to the
        distribution over possible answers.  
        """
        # Story Input Encoder
        story_embeddings = tf.nn.embedding_lookup(self.E, self.S)             # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.multiply(story_embeddings, self.story_mask)     # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.reduce_sum(story_embeddings, axis=[2])          # Shape: [None, story_len, embed_sz]

        # Query Input Encoder
        query_embedding = tf.nn.embedding_lookup(self.E, self.Q)              # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.multiply(query_embedding, self.query_mask)       # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.reduce_sum(query_embedding, axis=[1])            # Shape: [None, embed_sz]

        # Send Story through Memory Cell
        initial_state = self.cell.zero_state(self.bsz, dtype=tf.float32)
        _, memories = tf.nn.dynamic_rnn(self.cell, story_embeddings, sequence_length=self.S_len, 
                                        initial_state=initial_state)

        # Output Module 
        stacked_memories = tf.stack(memories, axis=1)
        
        # Generate Memory Scores
        p_scores = softmax(tf.reduce_sum(tf.multiply(stacked_memories,        # Shape: [None, mem_slots]
                                                     tf.expand_dims(query_embedding, 1)), axis=[2]))
        
        # Subtract max for numerical stability (softmax is shift invariant)
        p_max = tf.reduce_max(p_scores, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(p_scores - p_max)       
        attention = tf.expand_dims(attention, 2)                              # Shape: [None, mem_slots, 1]

        # Weight memories by attention vectors
        u = tf.reduce_sum(tf.multiply(stacked_memories, attention), axis=1)   # Shape: [None, embed_sz]

        # Output Transformations => Logits
        hidden = prelu(tf.matmul(u, self.H) + query_embedding)                # Shape: [None, embed_sz]
        logits = tf.matmul(hidden, self.R)                                    # Shape: [None, vocab_sz]
        
        return logits

    def loss(self):
        """
        Build loss computation - softmax cross-entropy between logits, and correct answer. 
        """
        return tf.losses.sparse_softmax_cross_entropy(self.A, self.logits)
    
    def train(self):
        """
        Build ADAM Optimizer Training Operation.
        """
        learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps, 
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, 
                                                   learning_rate=learning_rate, optimizer="Adam",
                                                   clip_gradients=self.clip_gradients)
        return train_op

class DynamicMemory(tf.contrib.rnn.RNNCell):
    def __init__(self, memory_slots, memory_size, keys, activation=prelu,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Instantiate a DynamicMemory Cell, with the given number of memory slots, and key vectors.

        :param memory_slots: Number of memory slots to initialize. 
        :param memory_size: Dimensionality of memories => tied to embedding size. 
        :param keys: List of keys to seed the Dynamic Memory with (can be random).
        :param initializer: Variable Initializer for Cell Parameters.
        """ 
        self.m, self.mem_sz, self.keys = memory_slots, memory_size, keys
        self.activation, self.init = activation, initializer

        # Instantiate Dynamic Memory Parameters => CONSTRAIN HERE
        self.U = tf.get_variable("U", [self.mem_sz, self.mem_sz], initializer=self.init)
        self.V = tf.get_variable("V", [self.mem_sz, self.mem_sz], initializer=self.init)
        self.W = tf.get_variable("W", [self.mem_sz, self.mem_sz], initializer=self.init)
    
    @property
    def state_size(self):
        """
        Return size of DynamicMemory State - for now, just M x d. 
        """
        return [self.mem_sz for _ in range(self.m)]
    
    @property
    def output_size(self):
        return [self.mem_sz for _ in range(self.m)]
    
    def zero_state(self, batch_size, dtype):
        """
        Initialize Memory to start as Key Values
        """
        return [tf.tile(tf.expand_dims(key, 0), [batch_size, 1]) for key in self.keys]

    def __call__(self, inputs, state, scope=None):
        """
        Run the Dynamic Memory Cell on the inputs, updating the memories with each new time step.

        :param inputs: 2D Tensor of shape [bsz, mem_sz] representing a story sentence.
        :param states: List of length M, each with 2D Tensor [bsz, mem_sz] => h_j (starts as key).
        """
        new_states = []
        for block_id, h in enumerate(state):
            # Gating Function
            content_g = tf.reduce_sum(tf.multiply(inputs, h), axis=[1])                  # Shape: [bsz]
            address_g = tf.reduce_sum(tf.multiply(inputs, 
                                      tf.expand_dims(self.keys[block_id], 0)), axis=[1]) # Shape: [bsz]
            g = sigmoid(content_g + address_g)

            # New State Candidate
            h_component = tf.matmul(h, self.U)                                           # Shape: [bsz, mem_sz]
            w_component = tf.matmul(tf.expand_dims(self.keys[block_id], 0), self.V)      # Shape: [1, mem_sz]
            s_component = tf.matmul(inputs, self.W)                                      # Shape: [bsz, mem_sz]
            candidate = self.activation(h_component + w_component + s_component)         # Shape: [bsz, mem_sz]

            # State Update
            new_h = h + tf.multiply(tf.expand_dims(g, -1), candidate)                    # Shape: [bsz, mem_sz]

            # Unit Normalize State 
            new_h_norm = tf.nn.l2_normalize(new_h, -1)                                   # Shape: [bsz, mem_sz]
            new_states.append(new_h_norm)
        
        return new_states, new_states