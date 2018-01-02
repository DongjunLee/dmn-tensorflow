
from tf.nn.rnn_cell importRNNCell


class AttnGRUCell(RNNCell):
    """Attention based GRU (cf. https://arxiv.org/abs/1603.01417).

    * Args:
        num_units: int, The number of units in the AttnGRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.  If not `True`, and the existing scope already has
         the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
         projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
  """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):

        super(AttnGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Attention Based GRU with num units cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)

    with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c
        return new_h, new_h
