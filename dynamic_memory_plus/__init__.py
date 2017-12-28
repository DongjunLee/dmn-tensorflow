
from hbconfig import Config
import tensorflow as tf


from .input import TextualInput


class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self,
              input=None,
              input_mask=None,
              question=None):

        facts = self._build_textual_input_module(input)
        encoded_question = self._build_question_module(question)
        last_memory = self._build_episodic_memory_module(facts, question)
        return self._build_question_module(last_memory)

    def _build_textual_input_module(self, input):
        textual_input = TextualInput(embed_dim=Config.model.embed_dim,
                                     vocab_size=Config.data.vocab_size,
                                     dtype=self.dtype)
        facts = textual_input.build(input)
        return facts

    def _build_question_module(self, question):
        encoder = Encoder(...)
        _, question = encoder.build(question)
        return question[0]

    def _build_episodic_memory_module(self, facts, question):
        # attention mechanism (gate attention + AttnGRU)
        # memory update
        pass

    def _build_answer_module(self):
        pass
