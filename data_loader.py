"""
bAbi data_loader
Original code : https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/utils.py
"""

import os as os
import numpy as np
from tqdm import tqdm



class DataLoader:

    def __init__(self, task_id, task_test_id, w2v_dim=100, input_mask_mode="sentence", use_pretrained=True):
        self.base_path = os.path.join("data/")

        self.task_id = str(task_id)
        self.task_test_id = str(task_test_id)
        self.w2v_dim = w2v_dim
        self.input_mask_mode = input_mask_mode
        self.use_pretrained = use_pretrained

    def make_train_and_test_set(self):
        train_raw, test_raw = self.get_babi_raw(self.task_id, self.task_test_id)
        self.max_facts_seq_len, self.max_question_seq_len, self.max_input_mask_len = self.get_max_seq_length(train_raw, test_raw)

        if self.use_pretrained:
            self.word2vec = self.load_glove(self.w2v_dim)
        else:
            self.word2vec = {}
        self.vocab = {}
        self.ivocab = {}

        self.create_vector("unknown")

        train_input, train_question, train_answer, train_input_mask = self.process_input(train_raw)
        test_input, test_question, test_answer, test_input_mask = self.process_input(test_raw)

        return {
            "train": (train_input, train_input_mask, train_question, train_answer),
            "test": (test_input, test_input_mask, test_question, test_answer)
        }

    def get_max_seq_length(self, *datasets):
        max_facts_length, max_question_length, max_input_mask_length = 0, 0, 0

        def count_punctuation(facts):
            return len(list(filter(lambda x: x == ".", facts)))

        for dataset in datasets:
            for d in dataset:
                max_facts_length = max(max_facts_length, len(d['C'].split()))
                max_input_mask_length = max(max_input_mask_length, count_punctuation(d['C']))
                max_question_length = max(max_question_length, len(d['Q'].split()))
        return max_facts_length, max_question_length, max_input_mask_length

    def init_babi(self, fname):
        print("==> Loading test from %s" % fname)
        tasks = []
        task = None
        for i, line in enumerate(open(fname)):
            id = int(line[0:line.find(' ')])
            if id == 1:
                task = {"C": "", "Q": "", "A": ""}

            line = line.strip()
            line = line.replace('.', ' . ')
            line = line[line.find(' ')+1:]
            if line.find('?') == -1:
                task["C"] += line
            else:
                idx = line.find('?')
                tmp = line[idx+1:].split('\t')
                task["Q"] = line[:idx]
                task["A"] = tmp[1].strip()
                tasks.append(task.copy())

        return tasks


    def get_babi_raw(self, id, test_id):
        babi_map = {
            "1": "qa1_single-supporting-fact",
            "2": "qa2_two-supporting-facts",
            "3": "qa3_three-supporting-facts",
            "4": "qa4_two-arg-relations",
            "5": "qa5_three-arg-relations",
            "6": "qa6_yes-no-questions",
            "7": "qa7_counting",
            "8": "qa8_lists-sets",
            "9": "qa9_simple-negation",
            "10": "qa10_indefinite-knowledge",
            "11": "qa11_basic-coreference",
            "12": "qa12_conjunction",
            "13": "qa13_compound-coreference",
            "14": "qa14_time-reasoning",
            "15": "qa15_basic-deduction",
            "16": "qa16_basic-induction",
            "17": "qa17_positional-reasoning",
            "18": "qa18_size-reasoning",
            "19": "qa19_path-finding",
            "20": "qa20_agents-motivations",
            "MCTest": "MCTest",
            "19changed": "19changed",
            "joint": "all_shuffled",
            "sh1": "../shuffled/qa1_single-supporting-fact",
            "sh2": "../shuffled/qa2_two-supporting-facts",
            "sh3": "../shuffled/qa3_three-supporting-facts",
            "sh4": "../shuffled/qa4_two-arg-relations",
            "sh5": "../shuffled/qa5_three-arg-relations",
            "sh6": "../shuffled/qa6_yes-no-questions",
            "sh7": "../shuffled/qa7_counting",
            "sh8": "../shuffled/qa8_lists-sets",
            "sh9": "../shuffled/qa9_simple-negation",
            "sh10": "../shuffled/qa10_indefinite-knowledge",
            "sh11": "../shuffled/qa11_basic-coreference",
            "sh12": "../shuffled/qa12_conjunction",
            "sh13": "../shuffled/qa13_compound-coreference",
            "sh14": "../shuffled/qa14_time-reasoning",
            "sh15": "../shuffled/qa15_basic-deduction",
            "sh16": "../shuffled/qa16_basic-induction",
            "sh17": "../shuffled/qa17_positional-reasoning",
            "sh18": "../shuffled/qa18_size-reasoning",
            "sh19": "../shuffled/qa19_path-finding",
            "sh20": "../shuffled/qa20_agents-motivations",
        }
        if (test_id == ""):
            test_id = id
        babi_name = babi_map[id]
        babi_test_name = babi_map[test_id]
        babi_train_raw = self.init_babi(os.path.join(self.base_path, 'en-10/%s_train.txt' % babi_name))
        babi_test_raw = self.init_babi(os.path.join(self.base_path, 'en-10/%s_test.txt' % babi_test_name))
        return babi_train_raw, babi_test_raw

    def load_glove(self, dim):
        word2vec = {}

        print("==> loading glove")
        with open(os.path.join(self.base_path, "glove/glove.6B." + str(dim) + "d.txt")) as f:
            for line in tqdm(f):
                l = line.split()
                word2vec[l[0]] = l[1:]

        print("==> glove is loaded")

        return word2vec

    def create_vector(self, word, silent=False):
        # if the word is missing from Glove, create some fake vector and store in glove!
        vector = np.random.uniform(0.0, 1.0, (self.w2v_dim,))
        self.word2vec[word] = vector
        if (not silent):
            print("data_loader.py::create_vector => %s is missing" % word)
        return vector

    def process_word(self, word, to_return="word2vec", silent=False):
        if not word in self.word2vec:
            self.create_vector(word, silent=silent)
        if not word in self.vocab:
            next_index = len(self.vocab)
            self.vocab[word] = next_index
            self.ivocab[next_index] = word

        if to_return == "word2vec":
            return self.word2vec[word]
        elif to_return == "index":
            return self.vocab[word]
        else:
            raise ValueError("return type is 'word2vec' or 'index'")

    def get_norm(self, x):
        x = np.array(x)
        return np.sum(x * x)

    def process_input(self, data_raw):
        questions = []
        inputs = []
        answers = []
        input_masks = []

        for x in data_raw:
            inp = x["C"].lower().split(' ')
            inp = [w for w in inp if len(w) > 0]

            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]

            inp_vector = [self.process_word(word=w, to_return="word2vec") for w in inp]
            inp_vector = self.pad_input(inp_vector, self.max_facts_seq_len, [np.zeros(self.w2v_dim)])

            q_vector = [self.process_word(word=w, to_return="word2vec") for w in q]
            q_vector = self.pad_input(q_vector, self.max_question_seq_len, [np.zeros(self.w2v_dim)])

            inputs.append(np.vstack(inp_vector).astype(float))
            questions.append(np.vstack(q_vector).astype(float))
            answers.append(self.process_word(word = x["A"], to_return = "index"))

            if self.input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32))
            elif self.input_mask_mode == 'sentence':
                input_mask = [index for index, w in enumerate(inp) if w == '.']
                input_mask = self.pad_input(input_mask, self.max_input_mask_len, [0])
                input_masks.append(input_mask)
            else:
                raise ValueError("input_mask_mode is only available (word, sentence)")

        return (np.array(inputs, dtype=np.float32),
                np.array(questions, dtype=np.float32),
                np.array(answers, dtype=np.int32).reshape(-1, 1),
                np.array(input_masks, dtype=np.int32))

    def pad_input(self, input_, size, pad_item):
        return input_ + pad_item * (size - len(input_))
