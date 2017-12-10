
from hbconfig import Config
import tensorflow as tf

from data_loader import DataLoader
import dataset
from model import DMN
import hook



def experiment_fn(run_config, params):

    dmn_model = DMN()
    estimator = tf.estimator.Estimator(
            model_fn=dmn_model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)

    data_loader = DataLoader(
            task_path=Config.data.task_path,
            task_id=Config.data.task_id,
            task_test_id=Config.data.task_id,
            w2v_dim=Config.model.embed_dim,
            use_pretrained=Config.model.use_pretrained)

    data = data_loader.make_train_and_test_set()

    vocab = data_loader.vocab

    # setting data property
    Config.data.vocab_size = len(vocab)
    Config.data.max_facts_seq_len = data_loader.max_facts_seq_len
    Config.data.max_question_seq_len = data_loader.max_question_seq_len
    Config.data.max_input_mask_length = data_loader.max_input_mask_len
    Config.eval.batch_size = len(data["test"][3])

    train_input_fn, train_input_hook = dataset.get_inputs(
            data["train"], batch_size=Config.train.batch_size, scope="train")
    test_input_fn, test_input_hook = dataset.get_inputs(
            data["test"], batch_size=Config.eval.batch_size, scope="test")

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=[
            train_input_hook],
        eval_hooks=[test_input_hook]
    )
    return experiment
