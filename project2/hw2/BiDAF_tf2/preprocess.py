import numpy as np
import data_io as pio
import re
import nltk
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        # char级别的set和word级别的set
        self.charset = set()
        self.word_set = set()
        self.build_charset()
        self.build_word_set()
        # 基于glove的预训练词向量
        self.wv_matrix = self.get_word2vec() 

    # 建立基于char级别的char to id和id to char的两个字典
    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)
            self.word_set |= self.dataset_word_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        # print(self.ch2id, self.id2ch)

    # 建立基于word级别的word to id和id to word的两个字典
    def build_word_set(self):
        for fp in self.datasets_fp:
            self.word_set |= self.dataset_word_info(fp)

        self.word_set = sorted(list(self.word_set))
        self.word_set = ['[PAD]', '[CLS]', '[SEP]'] + self.word_set + ['[UNK]']
        idx = list(range(len(self.word_set)))
        self.w2id = dict(zip(self.word_set, idx))
        self.id2w = dict(zip(idx, self.word_set))
        # print(self.w2id, self.id2w)
        print(len(self.w2id))

    # 基于数据，建立字符级别的set
    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset

    # 基于数据，建立词级别的set
    def dataset_word_info(self, inn):
        word_set = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            # 用nltk的word_tokenize来对句子做词语的切分
            word_set |= set(word_tokenize(context)) | set(word_tokenize(question)) | set(word_tokenize(answer))
            # charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return word_set

    # 从数据中提取context,question, text等
    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start


    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    # 把词转换为id
    def convert_word2id(self, sent, maxlen=None, begin=False, end=False):
        # 用nltk的word_tokenize来对句子做词语的切分
        w = [w for w in word_tokenize(sent)]
        w = ['[CLS]'] * begin + w

        if maxlen is not None:
            w = w[:maxlen - 1 * end]
            w += ['[SEP]'] * end
            w += ['[PAD]'] * (maxlen - len(w))
        else:
            w += ['[SEP]'] * end

        ids = list(map(self.get_word_id, w))

        return ids
    # 把字转换为id
    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids

    # 获取char对应的id，找不到的返回unk的id
    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    # 获取word对应的id，找不到的返回unk的id
    def get_word_id(self, w):
        return self.w2id.get(w, self.w2id['[UNK]'])

    # 生成char级别的数据
    def get_dataset(self, ds_fp):
        cs, qs, be = [], [], []
        for _, c, q, b, e in self.get_data(ds_fp):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))

    # 生成word级别的数据
    def get_dataset_word(self, ds_fp):
        cs, qs, be = [], [], []
        for _, c, q, b, e in self.get_word_data(ds_fp):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))

    # 获取char级别的数据
    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e

    # 获取word级别的数据
    def get_word_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids = self.get_sent_word_ids(context, self.max_clen)
            qids = self.get_sent_word_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(word_tokenize(text))
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e

    # 获取句子里的char级别的id
    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)

    # 获取句子里的word级别的id
    def get_sent_word_ids(self, sent, maxlen):
        return self.convert_word2id(sent, maxlen=maxlen, end=True)

    def get_word2vec(self):
        # 使用gensim来加载glove词向量
        glove_file = datapath('/Users/hyt/Workspaces/kaikeba/mingqi/project2/homework_02_code/BiDAF_tf2/data/glove.6B.100d.txt')
        word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
        glove2word2vec(glove_file, word2vec_glove_file)

        wv = KeyedVectors.load_word2vec_format(word2vec_glove_file)

        # 词向量初始化用随机向量
        word2vec_matrix = np.random.uniform(size=(len(self.w2id), 100))
        # oov_vector = np.random.uniform(size=(1, 100))[0]
        count = 0
        for word, index in self.w2id.items():
            # 如果在glove向量里，则把对应词的词向量替换掉随机向量
            if word.lower() in wv.vocab:
                word2vec_matrix[index] = wv[word.lower()]
            # 不在glove向量里，则保持随机向量
            else:
                count += 1
        print(count, 'unkown')
        return word2vec_matrix

 

if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    # print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
    # print(len(p.charset))
    # # print(p.word_set)
    # print(len(p.word_set))
    print(p.wv_matrix)