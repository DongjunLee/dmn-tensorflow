
from hbconfig import Config
import tensorflow as tf



class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def get_inputs(data, buffer_size=10000, batch_size=64, scope="train"):

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):

            nonlocal data
            inputs, input_masks, questions, answers = data

            # Define placeholders
            input_placeholder = tf.placeholder(
                tf.float32, [None, Config.data.max_facts_seq_len, Config.model.embed_dim])
            input_mask_placeholder = tf.placeholder(
                tf.int32, [None, None])
            question_placeholder = tf.placeholder(
                tf.float32, [None, Config.data.max_question_seq_len, Config.model.embed_dim])
            answer_placeholder = tf.placeholder(
                tf.int32, [None, 1])

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_placeholder, input_mask_placeholder,
                 question_placeholder, answer_placeholder))

            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_input, next_input_mask, next_question, next_answer = iterator.get_next()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: inputs,
                               input_mask_placeholder: input_masks,
                               question_placeholder: questions,
                               answer_placeholder: answers})

            # Return batched (features, labels)
            features = {"input_data": next_input,
                        "input_data_mask": next_input_mask,
                        "question_data": next_question}
            return (features, next_answer)

    # Return function and hook
    return inputs, iterator_initializer_hook
