
from hbconfig import Config
import tensorflow as tf


__all__ = [
    "Encoder", "Episode", "AttentionGate"
]



class Encoder:
    """Encoder class is Mutil-layer Recurrent Neural Networks

    The 'Encoder' usually encode the sequential input vector.
    """

    UNI_ENCODER_TYPE = "UNI"
    BI_ENCODER_TYPE = "BI"

    RNN_GRU_CELL = "GRU"
    RNN_LSTM_CELL = "LSTM"
    RNN_LAYER_NORM_LSTM_CELL = "LAYER_NORM_LSTM"
    RNN_NAS_CELL = "NAS"

    def __init__(self, encoder_type="UNI", num_layers=4,
                 cell_type="GRU", num_units=512, dropout=0.8,
                 dtype=tf.float32):
        """Contructs an 'Encoder' instance.

        * Args:
            encoder_type: rnn encoder_type (UNI, BI)
            num_layers: number of RNN cell composed sequentially of multiple simple cells.
            input_vector: RNN Input vectors.
            sequence_length: batch element's sequence length
            cell_type: RNN cell types (LSTM, GRU, LAYER_NORM_LSTM, NAS)
            num_units: the number of units in cell
            dropout: set prob operator adding dropout to inputs of the given cell.
            dtype: the dtype of the input

        * Returns:
            Encoder instance
        """

        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.num_units = num_units
        self.dropout = dropout
        self.dtype = dtype

    def build(self, input_vector, sequence_length, scope=None):
        if self.encoder_type == self.UNI_ENCODER_TYPE:
            self.cells = self._create_rnn_cells()

            return self.unidirectional_rnn(input_vector, sequence_length, scope=scope)
        elif self.encoder_type == self.BI_ENCODER_TYPE:
            self.cells_fw = self._create_rnn_cells(is_list=True)
            self.cells_bw = self._create_rnn_cells(is_list=True)

            return self.bidirectional_rnn(input_vector, sequence_length, scope=scope)
        else:
            raise ValueError(f"Unknown encoder_type {encoder_type}")

    def unidirectional_rnn(self, input_vector, sequence_length, scope=None):
        return tf.nn.dynamic_rnn(
                self.cells,
                input_vector,
                sequence_length=sequence_length,
                dtype=self.dtype,
                time_major=False,
                swap_memory=True,
                scope=scope)

    def bidirectional_rnn(self, input_vector, sequence_length, scope=None):
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                self.cells_fw,
                self.cells_bw,
                input_vector,
                sequence_length=sequence_length,
                dtype=self.dtype,
                scope=scope)

        encoder_final_state = tf.concat((output_state_fw[-1], output_state_bw[-1]), axis=1)
        return outputs, encoder_final_state

    def _create_rnn_cells(self, is_list=False):
        """Contructs stacked_rnn with num_layers

        * Args:
            is_list: flags for stack bidirectional. True=stack bidirectional, False=unidirectional

        * Returns:
            stacked_rnn
        """

        stacked_rnn = []
        for _ in range(self.num_layers):
            single_cell = self._rnn_single_cell()
            stacked_rnn.append(single_cell)

        if is_list:
            return stacked_rnn
        else:
            return tf.nn.rnn_cell.MultiRNNCell(
                    cells=stacked_rnn,
                    state_is_tuple=True)

    def _rnn_single_cell(self):
        """Contructs rnn single_cell"""

        if self.cell_type == self.RNN_GRU_CELL:
            single_cell = tf.contrib.rnn.GRUCell(
                self.num_units,
                reuse=tf.get_variable_scope().reuse)
        elif self.cell_type == self.RNN_LSTM_CELL:
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                self.num_units,
                forget_bias=1.0,
                reuse=tf.get_variable_scope().reuse)
        elif self.cell_type == self.RNN_LAYER_NORM_LSTM_CELL:
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units,
                forget_bias=1.0,
                layer_norm=True,
                reuse=tf.get_variable_scope().reuse)
        elif self.cell_type == self.RNN_NAS_CELL:
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units)
        else:
            raise ValueError(f"Unknown rnn cell type. {self.cell_type}")

        if self.dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - self.dropout))

        return single_cell


class Episode:
    """Episode class is used update memory in in Episodic Memory Module"""

    def __init__(self, num_units):
        self.gate = AttentionGate(hidden_size=num_units)
        self.rnn = tf.contrib.rnn.GRUCell(num_units)

    def update(self, c, m_t, q_t):
        """Update memory with attention mechanism

        * Args:
            c : encoded raw text and stacked by each sentence
                shape: fact_count x [batch_size, num_units]
            m_t : previous memory
                shape: [num_units, batch_size]
            q_t : encoded question last state
                shape: [num_units, batch_size]

        * Returns:
            h : updated memory
        """
        h = tf.zeros_like(c[0])

        with tf.variable_scope('memory-update') as scope:
            for fact in c:
                g = self.gate.score(tf.transpose(fact, name="c"), m_t, q_t)
                print("g: ", g)
                h = g * self.rnn(fact, h, scope="episode_rnn")[0] + (1 - g) * h
                scope.reuse_variables()
        return h


class AttentionGate:
    """AttentionGate class is simple two-layer feed forward neural network with Score function."""

    def __init__(self, hidden_size=4):
        self.w1 = tf.get_variable(
                "w1", [hidden_size, 7*hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(Config.model.reg_scale))
        self.b1 = tf.get_variable("b1", [hidden_size, 1])
        self.w2 = tf.get_variable(
                "w2", [1, hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(Config.model.reg_scale))
        self.b2 = tf.get_variable("b2", [1, 1])

    def score(self, c_t, m_t, q_t):
        """For captures a variety of similarities between input(c), memory(m) and question(q)

        * Args:
            c_t : transpose of one fact (encoded sentence's last state)
                  shape: [num_units, batch_size]
            m_t : transpose of previous memory
                  shape: [num_units, batch_size]
            q_t : transpose of encoded question
                  shape: [num_units, batch_size]

        * Returns:
            gate score
            shape: [batch_size, 1]
        """

        with tf.variable_scope('attention_gate'):
            z = tf.concat([c_t, m_t, q_t, c_t*q_t, c_t*m_t, (c_t-q_t)**2, (c_t-m_t)**2], 0)

            tf.nn.tanh(tf.matmul(self.w1, z) + self.b1)
            o1 = tf.nn.tanh(tf.matmul(self.w1, z) + self.b1)
            o2 = tf.nn.sigmoid(tf.matmul(self.w2, o1) + self.b2)
            return tf.transpose(o2)
