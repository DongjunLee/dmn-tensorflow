
from hbconfig import Config
import tensorflow as tf

from .encoder import Encoder
from .episode import Episode



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self,
              embedding_input=None,
              input_mask=None,
              embedding_question=None):

        facts, question = self._build_input_module(embedding_input, input_mask, embedding_question)
        last_memory = self._build_episodic_memory(facts, question)
        return self._build_answer_decoder(last_memory)

    def _build_input_module(self, embedding_input, input_mask, embedding_question):
        encoder = Encoder(
            encoder_type=Config.model.encoder_type,
            num_layers=Config.model.num_layers,
            cell_type=Config.model.cell_type,
            num_units=Config.model.num_units,
            dropout=Config.model.dropout)

        # slice zeros padding
        input_length = tf.reduce_max(input_mask, axis=1)
        question_length = tf.reduce_sum(tf.to_int32(
            tf.not_equal(tf.reduce_max(embedding_question, axis=2), Config.data.PAD_ID)), axis=1)

        with tf.variable_scope("input-module") as scope:
            input_encoder_outputs, _ = encoder.build(
                    embedding_input, input_length, scope="encoder")

            with tf.variable_scope("facts") as scope:
                batch_size = tf.shape(input_mask)[0]
                max_mask_length = tf.shape(input_mask)[1]

                def get_encoded_fact(i):
                    nonlocal input_mask

                    mask_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input_mask[i], Config.data.PAD_ID)), axis=0)
                    input_mask = tf.boolean_mask(input_mask[i], tf.sequence_mask(mask_lengths, max_mask_length))

                    encoded_facts = tf.gather_nd(input_encoder_outputs[i], tf.reshape(input_mask, [-1, 1]))
                    padding = tf.zeros(tf.stack([max_mask_length - mask_lengths, Config.model.num_units]))
                    return tf.concat([encoded_facts, padding], 0)

                facts_stacked = tf.map_fn(get_encoded_fact, tf.range(start=0, limit=batch_size), dtype=self.dtype)

                # max_input_mask_length x [batch_size, num_units]
                facts = tf.unstack(tf.transpose(facts_stacked, [1, 0, 2]), num=Config.data.max_input_mask_length)

        with tf.variable_scope("input-module") as scope:
            scope.reuse_variables()
            _, question = encoder.build(
                    embedding_question, question_length, scope="encoder")

        return facts, question[0]


    def _build_episodic_memory(self, facts, question):

        with tf.variable_scope('episodic-memory-module') as scope:
            memory = tf.identity(question)

            episode = Episode(Config.model.num_units, reg_scale=Config.model.reg_scale)
            rnn = tf.contrib.rnn.GRUCell(Config.model.num_units)

            for _ in range(Config.model.memory_hob):
                updated_memory = episode.update(facts,
                        tf.transpose(memory, name="m"),
                        tf.transpose(question, name="q"))
                memory, _ = rnn(updated_memory, memory, scope="memory_rnn")
                scope.reuse_variables()
        return memory

    def _build_answer_decoder(self, last_memory):

        with tf.variable_scope('answer-module'):
            w_a = tf.get_variable(
                    "w_a", [Config.model.num_units, Config.data.vocab_size],
                    regularizer=tf.contrib.layers.l2_regularizer(Config.model.reg_scale))
            logits = tf.matmul(last_memory, w_a)
        return logits
