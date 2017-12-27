#-- coding: utf-8 -*-

import argparse
import logging

from hbconfig import Config
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from data_loader import DataLoader
import hook
from model import Model


def experiment_fn(run_config, params):

    model = Model()
    estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
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
    print("max_facts_seq_len:", data_loader.max_facts_seq_len)
    print("max_question_seq_len:", data_loader.max_question_seq_len)
    print("max_input_mask_length:", data_loader.max_input_mask_len)

    train_input_fn, train_input_hook = data_loader.make_batch(
            data["train"], batch_size=Config.model.batch_size, scope="train")
    test_input_fn, test_input_hook = data_loader.make_batch(
            data["test"], batch_size=Config.model.batch_size, scope="test")

    train_hooks = [train_input_hook]
    if Config.train.print_verbose:
        pass
    if Config.train.debug:
        train_hooks.append(tf_debug.LocalCLIDebugHook())

    eval_hooks = [test_input_hook]
    if Config.train.debug:
        eval_hooks.append(tf_debug.LocalCLIDebugHook())

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=train_hooks,
        eval_hooks=eval_hooks
    )
    return experiment


def main(mode):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    run_config = tf.contrib.learn.RunConfig(
            model_dir=Config.train.model_dir,
            save_checkpoints_steps=Config.train.save_checkpoints_steps)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=mode,
        hparams=params
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    args = parser.parse_args()

    tf.logging._logger.setLevel(logging.INFO)

    Config(args.config)
    print("Config: ", Config)
    if Config.description:
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main(args.mode)
