
import tensorflow as tf



class Episode:

    def __init__(self, num_units):
        self.gate = AttentionGate()
        self.attn_gru = self._build_attention_based_gru()
        self.rnn = self._build_attention_based_gru(num_units)

    def _build_attention_based_gru(self, num_units):
        pass

    def update(self, f, m_t, q_t):
        h = tf.zeros_like(f[0])

        with tf.variable_scope('memory-update') as scope:
            for fact in f:
                g = self.gate.score(tf.transpose(fact, name="f"), m_t, q_t)
                h = g * self.rnn(fact, h, scope="episode_rnn")[0] + (1 - g) * h
                scope.reuse_variables()
        return h


class AttentionGate:

    def __init__(self, hidden_size=4, reg_scale=0.001):
        self.w1 = tf.get_variable(
                "w1", [hidden_size, 7*hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
        self.b1 = tf.get_variable("b1", [hidden_size, 1])
        self.w2 = tf.get_variable(
                "w2", [1, hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
        self.b2 = tf.get_variable("b2", [1, 1])

    def score(self, f_t, m_t, q_t):

        with tf.variable_scope('attention_gate'):
            z = tf.concat([f_t * q_t, f_t * m_t, tf.abs(t_f - q_t), tf.abs(f_t - m_t)], axis=0)

            o1 = tf.nn.tanh(tf.matmul(self.w1, z) + self.b1)
            o2 = tf.matmul(self.w2, o1) + self.b2
            o3 = tf.softmax(o2)
            return tf.transpose(o3)
