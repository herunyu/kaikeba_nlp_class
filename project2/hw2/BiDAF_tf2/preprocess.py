import numpy as np
import data_io as pio
import re
import nltk
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm


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
        print('建立基于char级别的char to id和id to char的两个字典..')
        if 'charset.npy' in os.listdir('./cache'):
            print('已存在，读取charset..')
            self.charset = np.load('./cache/charset.npy')
            self.charset = self.charset.tolist()
            print('读取charset完毕')
        else:
            for fp in tqdm(self.datasets_fp):
                self.charset |= self.dataset_info(fp)
                # self.word_set |= self.dataset_word_info(fp)

            self.charset = sorted(list(self.charset))
            self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
            tmp_char_list = np.array(self.charset)
            np.save('./cache/charset.npy', tmp_char_list)

        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        # print(self.ch2id, self.id2ch)

    # 建立基于word级别的word to id和id to word的两个字典
    def build_word_set(self):
        print('建立基于word级别的word to id和id to word的两个字典..')
        if 'wordset.npy' in os.listdir('./cache'):
            print('已存在，读取wordset..')
            self.word_set = np.load('./cache/wordset.npy')
            self.word_set = self.word_set.tolist()
            print('读取wordset完毕')
        else:
            for fp in tqdm(self.datasets_fp):
                self.word_set |= self.dataset_word_info(fp)

            self.word_set = sorted(list(self.word_set))
            self.word_set = ['[PAD]', '[CLS]', '[SEP]'] + self.word_set + ['[UNK]']
            tmp_word_list = np.array(self.word_set)
            np.save('./cache/wordset.npy', tmp_word_list)

        idx = list(range(len(self.word_set)))
        self.w2id = dict(zip(self.word_set, idx))
        self.id2w = dict(zip(idx, self.word_set))

    # 基于数据，建立字符级别的set
    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in tqdm(self.iter_cqa(dataset)):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset

    # 基于数据，建立词级别的set
    def dataset_word_info(self, inn):
        word_set = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in tqdm(self.iter_cqa(dataset)):
            # 用nltk的word_tokenize来对句子做词语的切分
            word_set |= set(word_tokenize(context)) | set(word_tokenize(question)) | set(word_tokenize(answer))
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return word_set

    # 从数据中提取context,question, text等
    def iter_cqa(self, dataset):
        for data in tqdm(dataset['data']):
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
    def convert2id(self, sent, maxlen=None, c_maxlen=None, begin=False, end=False):
        # 用nltk的word_tokenize来对句子做词语的切分
        w = [w for w in word_tokenize(sent)]
        ch_list = []
        w = ['[CLS]'] * begin + w
        if maxlen is not None:
            w = w[:maxlen - 1 * end]
            w += ['[SEP]'] * end
            w += ['[PAD]'] * (maxlen - len(w))
        else:
            w += ['[SEP]'] * end   
        for c in w:
            if c == '[CLS]':
                ch_list.append(['[CLS]'] * c_maxlen)
            elif c == '[SEP]':
                ch_list.append(['[SEP]'] * c_maxlen)
            elif c == '[PAD]':
                ch_list.append(['[PAD]'] * c_maxlen)
            else:
                c = list(c[:c_maxlen]) + ['[PAD]'] * (c_maxlen - len(c))
                ch_list.append(c)

        word_ids = np.array([self.get_word_id(word) for word in w])
        ch_ids = [[self.get_id(ch) for ch in w ] for w in ch_list]
        
        return ch_ids, word_ids


    # 获取char对应的id，找不到的返回unk的id
    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    # 获取word对应的id，找不到的返回unk的id
    def get_word_id(self, w):
        return self.w2id.get(w, self.w2id['[UNK]'])

    # 生成char级别的数据
    def get_dataset(self, ds_fp):
        cs, qs, be = [], [], []
        for _, c, q, b, e in tqdm(self.get_data(ds_fp)):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))

    # 生成word级别的数据
    def get_dataset_word(self, ds_fp, data_type):
        cs, cws, qs, qws, be = [], [], [], [], []
        for _, c, q, b, e in tqdm(self.get_word_data(ds_fp, data_type)):
            cs.append(c[0])
            cws.append(c[1])
            qs.append(q[0])
            qws.append(q[1])
            be.append((b, e))
        # print(be[0].shape)
        # return (np.array(cs), np.array(cws), np.array(qs), np.array(qws), np.array(be))
        return map(np.array, (cs, cws, qs, qws, be))

    # 获取char级别的数据
    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        print('获取char级别的数据')
        for qid, context, question, text, answer_start in tqdm(self.iter_cqa(dataset)):
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e

    # 获取word级别的数据
    def get_word_data(self, ds_fp, data_type):
        dataset = pio.load(ds_fp)
        print('获取word级别的数据')
        return_list = []
        if 'word_data_{}.npy'.format(data_type) in os.listdir('./cache'):
            print('已存在，读取word_data_{}..'.format(data_type))
            return_list = np.load('./cache/word_data_{}.npy'.format(data_type), allow_pickle=True)
            return_list = return_list.tolist()
            print('读取word_data_{}完毕'.format(data_type))
        else:
            for qid, context, question, text, answer_start in tqdm(self.iter_cqa(dataset)):
                cids = self.get_sent_word_ids(context, self.max_clen)
                qids = self.get_sent_word_ids(question, self.max_qlen)
                b, e = answer_start, answer_start + len(word_tokenize(text))
                if e >= len(cids):
                    b = e = 0
                # yield qid, cids, qids, b, e
                return_list.append((qid, cids, qids, b, e))
            tmp_word_list = np.array(return_list)
            np.save('./cache/word_data_{}.npy'.format(data_type), tmp_word_list)
        
        return return_list

    # 获取句子里的char级别的id
    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)

    # 获取句子里的word级别的id
    def get_sent_word_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, c_maxlen=20, end=True)

    def get_word2vec(self):
        # 使用gensim来加载glove词向量
        print('加载glove词向量中....')
        if 'wv_matrix.npy' in os.listdir('./cache'):
            print('已存在，读取wv_matrix..')
            word2vec_matrix = np.load('./cache/wv_matrix.npy')
            # wv_matrix = wv_matrix.tolist()
            print('读取wv_matrix完毕')
        else:
            # glove_file = datapath('/Users/hyt/Workspaces/kaikeba/mingqi/project2/homework_02_code/BiDAF_tf2/data/glove.6B.100d.txt')
            glove_file = datapath('C:\\Users\\herunyu\\Desktop\\glove.6B.100d.txt')
            word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
            glove2word2vec(glove_file, word2vec_glove_file)

            wv = KeyedVectors.load_word2vec_format(word2vec_glove_file)
            print('词向量加载完毕.')
            # 词向量初始化用随机向量
            word2vec_matrix = np.random.uniform(size=(len(self.w2id), 100))
            # oov_vector = np.random.uniform(size=(1, 100))[0]
            count = 0
            for word, index in tqdm(self.w2id.items()):
                # 如果在glove向量里，则把对应词的词向量替换掉随机向量
                if word.lower() in wv.vocab:
                    word2vec_matrix[index] = wv[word.lower()]
                # 不在glove向量里，则保持随机向量
                else:
                    count += 1
            print(count, 'unkown')
            # tmp_word_list = np.array(self.word_set)
            np.save('./cache/wv_matrix.npy', word2vec_matrix)
        
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
    chids, wids = p.convert2id('How', 10,10, end=True)
    print(chids)
    print(type(chids))
    print(type(chids[0]))
    print(chids.shape)
    print(wids)
    print(wids.shape)