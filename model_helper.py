
import tensorflow as tf


__all__ = [
    "create_unidirectional_rnn", "create_bidirectional_rnn",
    "create_rnn_cells", "rnn_single_cell",
]


def create_unidirectional_rnn(cells, input_embed, sequence_length, dtype):
        return tf.nn.dynamic_rnn(
                cells,
                input_embed,
                sequence_length=sequence_length,
                dtype=dtype,
                time_major=False,
                swap_memory=True)


def create_bidirectional_rnn(cells_fw, cells_bw, input_embed, sequence_length):

    outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw,
            cells_bw,
            input_embed,
            sequence_length=sequence_length,
            dtype=tf.float32)

    encoder_final_state = tf.concat((output_state_fw[-1], output_state_bw[-1]), axis=1)

    return outputs, encoder_final_state


def create_rnn_cells(cell_type, dropout, num_units, num_layers, is_list=False):
    stacked_rnn = []
    for _ in range(num_layers):
        single_cell = rnn_single_cell(cell_type, dropout, num_units)
        stacked_rnn.append(single_cell)

    if is_list:
        return stacked_rnn
    else:
        return tf.nn.rnn_cell.MultiRNNCell(
                cells=stacked_rnn,
                state_is_tuple=True)


def rnn_single_cell(cell_type, dropout, num_units):
    if cell_type == "GRU":
        single_cell = tf.contrib.rnn.GRUCell(
            num_units)
    elif cell_type == "LSTM":
        single_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units,
            forget_bias=1.0)
    elif cell_type == "LAYER_NORM_LSTM":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=1.0,
            layer_norm=True)
    elif cell_type == "NAS":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units)
    else:
        raise ValueError(f"Unknown rnn cell type. {cell_type}")

    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

    return single_cell
