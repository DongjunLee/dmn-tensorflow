
import tensorflow as tf


__all__ = [
    "Encoder", "Episode", "AttentionGate"
]



class Encoder:

    UNI_ENCODER_TYPE = "UNI"
    BI_ENCODER_TYPE = "BI"

    RNN_GRU_CELL = "GRU"
    RNN_LSTM_CELL = "LSTM"
    RNN_LAYER_NORM_LSTM_CELL = "LAYER_NORM_LSTM"
    RNN_NAS_CELL = "NAS"

    def __init__(self, encoder_type="UNI", num_layers=4,
                 input_vector=None, sequence_length=None,
                 cell_type="GRU", num_units=512, dropout=0.8,
                 dtype=tf.float32):

        self.num_layers = num_layers
        self.input_vector = input_vector
        self.sequence_length = sequence_length
        self.cell_type = cell_type
        self.num_units = num_units
        self.dropout = dropout
        self.dtype = dtype

        if encoder_type == self.UNI_ENCODER_TYPE:
            return self.unidirectional_rnn()
        elif encoder_type == self.BI_ENCODER_TYPE:
            return self.bidirectional_rnn()
        else:
            raise ValueError(f"Unknown encoder_type {encoder_type}")


    def unidirectional_rnn(self):
        cells = self._create_rnn_cells()
        return tf.nn.dynamic_rnn(
                cells,
                self.input_vector,
                sequence_length=self.sequence_length,
                dtype=self.dtype,
                time_major=False,
                swap_memory=True)


    def bidirectional_rnn(self):

        cells_fw = self._create_rnn_cells(self.num_units, is_list=True)
        cells_bw = self._create_rnn_cells(self.num_units, is_list=True)

        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                self.input_vector,
                sequence_length=self.sequence_length,
                dtype=self.dtype)

        encoder_final_state = tf.concat((output_state_fw[-1], output_state_bw[-1]), axis=1)

        return outputs, encoder_final_state


    def _create_rnn_cells(self, is_list=False):
        stacked_rnn = []
        for _ in range(self.num_layers):
            single_cell = self._rnn_single_cell(self.cell_type, self.dropout, self.num_units)
            stacked_rnn.append(single_cell)

        if is_list:
            return stacked_rnn
        else:
            return tf.nn.rnn_cell.MultiRNNCell(
                    cells=stacked_rnn,
                    state_is_tuple=True)


    def _rnn_single_cell(self):
        if self.cell_type == self.RNN_GRU_CELL:
            single_cell = tf.contrib.rnn.GRUCell(
                self.num_units)
        elif self.cell_type == self.RNN_LSTM_CELL:
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                self.num_units,
                forget_bias=1.0)
        elif self.cell_type == self.RNN_LAYER_NORM_LSTM_CELL:
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units,
                forget_bias=1.0,
                layer_norm=True)
        elif self.cell_type == self.RNN_NAS_CELL:
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units)
        else:
            raise ValueError(f"Unknown rnn cell type. {self.cell_type}")

        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - self.dropout))

        return single_cell


class Episode:

    def __init__(self):
        self.gate = AttentionGate()
        self.rnn = model_helper.rnn_single_cell(Config.model.cell_type, Config.model.dropout, Config.model.num_units)

    def update(self, c, m, q):
        h = tf.zero_likes(shape)

        for c, c_t in zip(c, tf.transpose(c)):
            g = self.gate.score(c_t, m, q)
            h = g * self.rnn(c, h) + (1 - g) * h
        return h


class AttentionGate:
    # gate : G(c_t, m^i-1, q), 2-layer feed forward network layer

    def __init__(self):
        self.w1 = tf.Variable()
        self.b1 = tf.Variable()
        self.w2 = tf.Variable()
        self.b2 = tf.Variable()

    def score(c, m, q):
        with tf.variable_scope('attention_gate'):
            # for captures a variety of similarities between input(c), memory(m) and question(q)
            z = tf.concat([c, m, q, c*q, c*m, (c-q)**2, (c-m)**2, 0])

            tanh( self.w1 * z + self.b1 )
            o1 = tf.nn.tanh(tf.matmul(self.w1, z) + self.b1)
            o2 = tf.nn.sigmoid(tf.matmul(self.w2, o1) + self.b2)
            return o2
