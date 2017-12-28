
import tensorflow as tf



class Episode:

    def __init__(self, num_units):
        self.gate = AttentionGate()
        self.attn_gru = self._build_attention_based_gru()
        self.rnn = tf.contrib.rnn.GRUCell(num_units)

    def _build_attention_based_gru(self):
        pass

    def update(self):
        pass


class AttentionGate:

    def __init__(self):
        pass

    def score(self,):
        pass
