"""
entity_network.py

Model definition class for the Recurrent Entity Network. Defines Input Encoder,
Dynamic Memory Cell, and Readout components.
"""
from collections import namedtuple
import tensorflow as tf
import tensorflow.nn.rnn_cell_impl._RNNCell as RNNCell
from tflearn.activations import sigmoid, softmax, prelu

PAD_ID = 0

class EntityNetwork():
    def __init__(self, vocabulary, sentence_len, story_len, batch_size, embedding_size=100, 
                 memory_slots=20, initializer=tf.truncated_normal_initializer(stddev=0.1)):
        """
        Initialize an Entity Network with the necessary hyperparameters.

        :param vocabulary: Word Vocabulary for given model.
        :param sentence_len: Maximum length of a sentence.
        :param story_len: Maximum length of a story.
        """
        self.vocab_sz, self.sentence_len, self.story_len = len(vocabulary), max_sentence, story_len
        self.embed_sz, self.memory_slots, self.init = embedding_size, memory_slots, initializer
        self.bsz = batch_size

        # Setup Placeholders
        self.S = tf.placeholder(tf.int32, [None, self.story_len, self.sentence_len], name="Story")
        self.S_len = tf.placeholder(tf.int32, [None], name="Story_Length")
        self.Q = tf.placeholder(tf.int32, [None, self.sentence_len], name="Query")
        self.A = tf.placeholder(tf.int32, [None], name="Answer")

        # Instantiate Network Weights
        self.instantiate_weights()

        # Build Inference Pipeline
        self.logits = self.inference()
    
    def instantiate_weights(self):
        """
        Instantiate Network Weights, including all weights for the Input Encoder, Dynamic
        Memory Cell, as well as Output Decoder.
        """
        # Create Embedding Matrix, with 0 Vector for PAD_ID (0)
        self.E = tf.get_variable("Embedding", [self.vocab_sz, self.embed_sz], initializer=self.init)
        self.E *= tf.constant([0] + [1 for _ in range(self.vocab_sz - 1)], tf.float32, shape=[self.vocab_sz, 1])
        
        # Create Learnable Mask
        self.input_mask = tf.get_variable("Input_Mask", [self.sentence_len, self.embed_sz], initializer=self.init)

        # Create Memory Cell Keys [IF DESIRED - TIE KEYS HERE]
        self.keys = [tf.get_variable("Key_%d" % i, [self.embed_sz], initializer=self.init) for i in range(self.memory_slots)]
        
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
        story_embeddings = tf.multiply(story_embeddings, self.input_mask)     # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.reduce_sum(story_embeddings, axis=[2])          # Shape: [None, story_len, embed_sz]

        # Query Input Encoder
        query_embedding = tf.nn.embedding_lookup(self.E, self.Q)              # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.multipy(query_embedding, self.input_mask)        # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.reduce_sum(query_embedding, axis=[1])            # Shape: [None, embed_sz]

        # Send Story through Memory Cell
        initial_state = self.cell.zero_state(self.bsz, dtype=tf.float32)
        _, memories = tf.nn.dynamic_rnn(self.cell, story_embeddings, self.S_len, initial_state)

        # Output Module 
        scores = []
        for m in memories:
            score = tf.reduce_sum(tf.multiply(query_embedding, m), axis=[-1]) # Shape: [None]
            score = tf.expand_dims(score, axis=-1)                            # Shape: [None, 1]
            scores.append(score)
        
        # Generate Memory Scores
        p_scores = softmax(tf.concat(scores, axis=[1]))                       # Shape: [None, mem_slots]
        
        # Get Weighted Sum of Memories
        attention = tf.expand_dims(p_scores, axis=-1)                         # Shape: [None, mem_slots, 1]
        memory_stack = tf.stack(memories, axis=1)                             # Shape: [None, mem_slots, embed_sz]
        u = tf.reduce_sum(attention * memory_stack, axis=1)                   # Shape: [None, embed_sz]

        # Output Transformations => Logits
        hidden = prelu(tf.matmul(u, self.H) + query_embedding)                # Shape: [None, embed_sz]
        logits = tf.matmul(hidden, self.R)                                    # Shape: [None, vocab_sz]
        return logits


class DynamicMemory(RNNCell):
    def __init__(self, memory_slots, memory_size, keys, activation=prelu,
                 initializer=tf.truncated_normal_initializer(stddev=0.1)):
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

    def __call__(self, inputs, state, scope=None):
        """
        Run the Dynamic Memory Cell on the inputs, updating the memories with each new time step.

        :param inputs: 3D Tensor of shape [bsz, story_len, mem_sz] representing story sentences.
        :param states: List of length M, each with 2D Tensor [bsz, mem_sz] => h_j (starts as 0).
        """
        new_states = []
        for block_id, h in enumerate(state):
            # Gating Function
            content_g = tf.reduce_sum(tf.multiply(inputs, h), axis=[1])                  # Shape: [bsz]
            address_g = tf.reduce_sum(tf.multiply(inputs, 
                                      tf.expand_dims(0, self.keys[block_id])), axis=[1]) # Shape: [bsz]
            g = sigmoid(content_g + address_g)

            # New State Candidate
            h_component = tf.matmul(h, self.U)                                           # Shape: [bsz, mem_sz]
            w_component = tf.matmul(tf.expand_dims(0, self.keys[block_id]), self.V)      # Shape: [1, mem_sz]
            s_component = tf.matmul(s, self.W)                                           # Shape: [bsz, mem_sz]
            candidate = self.activation(h_component + w_component + s_component)         # Shape: [bsz, mem_sz]

            # State Update
            new_h = h + tf.multiply(tf.expand_dims(g, -1), candidate)                    # Shape: [bsz, mem_sz]

            # Unit Normalize State 
            new_h_norm = tf.nn.l2_normalize(new_h, -1)                                   # Shape: [bsz, mem_sz]
            new_states.append(new_h_norm)
        
        return new_states
