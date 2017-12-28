
import numpy as np
import tensorflow as tf

from .encoder import Encoder



class TextualInput:

    def __init__(self, embed_dim, vocab_size, dtype=tf.float32):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dtype = dtype

    def build(self, input):
        fs = self.build_sentence_reader(input)
        facts = self.build_input_fusion_layer(fs)
        return facts

    def build_sentence_reader(self, input):

        with tf.variable_scope("sentence reader"):
            # fi = j=1 ~ M sum of (i_j o w_j^i)
            pe = tf.nn.embedding_lookup(self._positional_encoding(num_of_words, dim), input)
            w = tf.nn.embedding_lookup(self._word_embedding(), input)
            return pe * w # element wise

    def _word_embedding(self, dtype=tf.float32):
        return tf.get_variable("word embedding",
                               [self.vocab_size, self.embed_dim], dtype)

    def _positional_encoding(self, num_of_words, dim, dtype=tf.float32):
        # M = num_of_words, K = dim
        pe = np.array(
            [[ (1 - j/M) - (k/d) * (1 - 2j/M) for j in M(num_of_words) ] for k in K (dim)]
        )
        return tf.convert_to_tensor(pe, dtype=dtype, name="positional encoding")

    def build_input_fusion_layer(self, f):
        encoder = Encoder() # bidirectional rnn
        facts = encoder.build()
        return facts
