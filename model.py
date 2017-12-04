from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers

from . import model_helper



class DMN:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self._init_placeholder(features, labels)
        self.build_graph()

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"prediction": self.prediction})
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.train_pred,
                loss=self.loss,
                train_op=self.train_op,
                eval_metric_ops={
                    "accuracy": tf.metrics.accuracy(
                        tf.argmax(self.targets, axis=1), self.predictions)
                }
            )

    def _init_placeholder(self, features, labels):
        self.input_data = features
        if type(features) == dict:
            self.input_data = features["input_data"]
            self.question_data = features["question_data"]

        self.targets = labels

    def build_graph(self):
        self._build_semantic_memory()
        self._build_input_module()
        self._build_episodic_memory()
        self._build_attention_function()
        self._build_answer_decoder()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss()
            self._build_optimizer()

    def _build_semantic_memory(self):
        with tf.variable_scope("embeddings", dtype=self.dtype) as scope:
            self.embedding = tf.get_variable(
                    "embedding",
                    [Config.data.vocab_size, Config.model.embed_dim],
                    self.dtype)

            self.embedding_input = tf.nn.embedding_lookup(
                self.embedding, self.input_data)

    def _build_input_module(self):
        with tf.variable_scope("input-module"):
            self.input_encoder_outputs, _ = self._build_encoder("input_sentences", "input_sequences_length")

        with tf.variable_scope("input-module", reuse=True):
            self.question_encoder_outputs, _ = self._build_encoder("question_sentence", "question_sequence_length")

    def _build_encoder(self, input_data, input_sequence):
        cells = model_helper.create_rnn_cells(
                cell_type=Config.model.cell_type,
                dropout=Config.model.dropout,
                num_units=Config.model.num_units,
                num_layers=Config.model.num_layers)

        return model_helper.create_unidirectional_rnn(
                cells, input_data,
                input_sequence, self.dtype)

    def _build_episodic_memory(self):
        # nested recurrent neural networks
        # memory = GRU(e^i, m^i-1), m^0 = q
        # episode => g^i_t GRU(ct, h^i_t-1) + (1 - g^i_t) h^i_t-1
        pass

    def _build_attention_function(self):
        # gate : G(c_t, m^i-1, q), 2-layer feed forward network layer

    def _build_answer_decoder(self):
        pass

    def _build_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(
                self.targets,
                self.output,
                scope="loss")

        self.train_pred = tf.argmax(self.output[0], name='train/pred_0')
        self.predictions = tf.argmax(self.output, axis=1)

    def _build_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'learning_rate'],
            name="train_op")
