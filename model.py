from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

import dynamic_memory



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self.loss, self.train_op, self.eval_metric_ops, self.predictions = None, None, None, None
        self._init_placeholder(features, labels)
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=self.predictions,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self._build_metric()
        )

    def _init_placeholder(self, features, labels):
        self.input_data = features
        if type(features) == dict:
            self.embedding_input = features["input_data"]
            self.input_mask = features["input_data_mask"]
            self.embedding_question = features["question_data"]

        self.targets = labels

    def build_graph(self):
        graph = dynamic_memory.Graph(self.mode)
        output = graph.build(embedding_input=self.embedding_input,
                             input_mask = self.input_mask,
                             embedding_question=self.embedding_question)
        self.predictions = tf.argmax(output, axis=1)

        self._build_loss(output)
        self._build_optimizer()

    def _build_loss(self, output):
        with tf.variable_scope('loss'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    self.targets,
                    output,
                    scope="cross-entropy")
            reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.loss = tf.add(cross_entropy, reg_term)

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'gradients', 'learning_rate'],
            name="train_op")

    def _build_metric(self):
        return {
            "accuracy": tf.metrics.accuracy(self.targets, self.predictions)
        }
