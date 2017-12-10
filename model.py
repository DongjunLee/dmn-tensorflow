from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers

from model_helper import Encoder
from model_helper import Episode



class DMN:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self._set_batch_size(mode)

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
                        self.targets, self.predictions)
                }
            )

    def _set_batch_size(self, mode):
        if mode == tf.estimator.ModeKeys.EVAL:
            Config.model.batch_size = Config.eval.batch_size
        else:
            Config.model.batch_size = Config.train.batch_size

    def _init_placeholder(self, features, labels):
        self.input_data = features
        if type(features) == dict:
            self.embedding_input = features["input_data"]

            self.input_mask = features["input_data_mask"]
            self.input_length = tf.reduce_max(self.input_mask, 1)
            self.embedding_question = features["question_data"]
            self.question_length = tf.map_fn(lambda x: tf.shape(x)[0],
                                             self.embedding_question,
                                             dtype=tf.int32)
        self.targets = labels

    def build_graph(self):
        self._build_input_module()
        self._build_episodic_memory()
        self._build_answer_decoder()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss()
            self._build_optimizer()

    def _build_input_module(self):
        encoder = self._build_encoder()

        with tf.variable_scope("input-module") as scope:
            self.input_encoder_outputs, _ = encoder.build()(
                    self.embedding_input, self.input_length, scope="encoder")

            self.facts = []
            max_mask_length = tf.shape(self.input_mask)[1]
            for i in range(Config.model.batch_size):
                input_mask = tf.identity(self.input_mask[i])
                mask_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input_mask, Config.data.PAD_ID)), 0)
                input_mask = tf.boolean_mask(input_mask, tf.sequence_mask(mask_lengths, max_mask_length))
                input_mask = tf.reshape(input_mask, [-1, 1])

                padding = tf.zeros(tf.stack([max_mask_length - mask_lengths, Config.model.num_units]))
                self.facts.append(tf.concat([tf.gather_nd(self.input_encoder_outputs[i], input_mask), padding], 0))

            facts_packed = tf.stack(self.facts)
            self.facts = tf.unstack(tf.transpose(facts_packed, [1, 0, 2]), num=Config.data.max_input_mask_length)

        with tf.variable_scope("input-module") as scope:
            scope.reuse_variables()
            _, self.question = encoder.build()(
                    self.embedding_question, self.question_length, scope="encoder")
            self.question = self.question[0]

    def _build_encoder(self):
        return Encoder(
                    encoder_type=Config.model.encoder_type,
                    num_layers=Config.model.num_layers,
                    cell_type=Config.model.cell_type,
                    num_units=Config.model.num_units,
                    dropout=Config.model.dropout)

    def _build_episodic_memory(self):
        with tf.variable_scope('episodic-memory-module') as scope:
            memory = tf.identity(self.question)

            episode = Episode(Config.model.num_units)
            rnn = tf.contrib.rnn.GRUCell(Config.model.num_units)

            for _ in range(Config.model.memory_hob):
                updated_memory = episode.update(self.facts,
                        tf.transpose(memory, name="m"),
                        tf.transpose(self.question, name="q"))
                memory, _ = rnn(updated_memory, memory, scope="memory_rnn")
                scope.reuse_variables()
            self.last_memory = memory


    def _build_answer_decoder(self):
        with tf.variable_scope('answer-module') as scope:
            w_a = tf.get_variable("w_a", [Config.model.num_units, Config.data.vocab_size])
            self.logits = tf.matmul(self.last_memory, w_a)

    def _build_loss(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy(
                self.targets,
                self.logits,
                scope="loss")

        self.train_pred = tf.argmax(self.logits[0], name='train/pred_0')
        self.predictions = tf.argmax(self.logits, axis=1)

    def _build_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss'],
            name="train_op")
