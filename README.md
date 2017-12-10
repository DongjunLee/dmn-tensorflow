# Dynamic Memory Network

TensorFlow implementation of [Ask Me Anything:
Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf).

![images](images/ask_me_anything_figure_3.png)


## Requirements

- Python 3.6
- TensorFlow 1.4
- hb-config
- tqdm

## Features

- Using Higher-APIs in TensorFlow
	- [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
	- [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)
	- [Dataset](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset)
- Dataset : [bAbl](https://research.fb.com/downloads/babi/)

## Todo




## Config

example: bAbi_task1.yml

```yml
data:
  base_path: 'data/'
  task_path: 'en/'
  task_id: 1
  PAD_ID: 0

model:
  use_pretrained: true  (true or false)
  embed_dim: 200  (if use_pretrained: only available 50, 100, 200, 300)
  encoder_type: UNI  ('UNI', 'BI')
  cell_type: GRU  (LSTM, GRU, LAYER_NORM_LSTM, NAS)
  num_layers: 3 
  num_units: 256
  memory_hob: 3
  dropout: 0.8

train:
  batch_size: 128
  learning_rate: 0.0001
  train_steps: 100000
  model_dir: 'logs/bAbi_task1'
  save_checkpoints_steps: 1000
  check_hook_n_iter: 1000
  min_eval_frequency: 1000
  optimizer: 'Adam'  ('Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD')

eval:
  batch_size: -1   (Using all test data)
```


## Usage

Install requirements.

```pip install -r requirements.txt```

Then, prepare dataset and pre-trained glove.

```
sh scripts/fetch_babi_data.sh
sh scripts/fetch_glove_data.sh
```

Finally, start trand and evalueate model
```
python main.py --config bAbi_task1 --mode train_and_evaluate
```

### Tensorboar

```tensorboard --logdir logs```


## Reference

- [Implementing Dynamic memory networks](https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/)
- [Ask Me Anything:
Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf) (2015) by A Kumar
