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

        # for pre-trained embedding (scaffold = object that can be used to set initialization, saver, and more to be used in training.)
        # tf.estimator.EstimatorSpec(..., scaffold=tf.train.Scaffold(init_feed_dict={embed_ph: my_embedding_numpy_array}

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
            self.input_encoder_outputs, _ = model_helper.Encoder(input_vector=None, sequence_length=None)
            # ... gather
            self.facts = None

        with tf.variable_scope("input-module", reuse=True):
            question_encoder_outputs, _ = model_helper.Encoder(input_vector=None, sequence_length=None)
            self.question = question_encoder_outputs[-1]

    def _build_episodic_memory(self):
        # nested recurrent neural networks
        with tf.variable_scope('episodic') as scope:
            memory = tf.identity(self.question)

            episode = model_helper.Episode()
            rnn = model_helper.rnn_single_cell(Config.model.cell_type, Config.model.dropout, Config.model.num_units)

            for _ in range(Config.model.memory_hob):
                memory, _  = rnn(episode.update(self.facts, memeory, self.question), memory)
            self.last_memory = memory


    def _build_answer_decoder(self):
        w_a = tf.Variable()
        a = tf.identity(self.last_memory)

        y = softmax(tf.matmul(w_a, a))
        a = gru(tf.concat([y, self.question], 0), a)

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
            summaries=['loss'],
            name="train_op")
