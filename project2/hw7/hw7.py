import json, os
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Layer, Dense, Permute
from keras.models import Model
from tqdm import tqdm

# 基本信息
maxlen = 128
epochs = 20
batch_size = 16
learing_rate = 2e-5

# 设置数据路径
data_dir='/home/hry/Workspace/kaikeba_nlp/dureader_robust-data'
output_dir='/home/hry/Workspace/kaikeba_nlp/output'

# 设置模型路径
bert_dir = '/home/hry/Workspace/kaikeba_nlp/roberta_zh_l12'
config_path = f'{bert_dir}/bert_config.json'
checkpoint_path = f'{bert_dir}/bert_model.ckpt'
dict_path = f'{bert_dir}/vocab.txt'

# 加载数据
def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],
                [a['text'] for a in qa.get('answers', [])]
            ])
    return D

# 读取数据
train_data = load_data(
    os.path.join(data_dir,'train.json')
    # os.path.join(data_dir,'demo_train.json')
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 子串搜索
def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

# 数据生成器
class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            context, question, answers = item[1:]
            token_ids, segment_ids = tokenizer.encode(
                question, context, maxlen=maxlen
            )
            a = np.random.choice(answers)
            a_token_ids = tokenizer.encode(a)[0][1:-1]
            start_index = search(a_token_ids, token_ids)
            if start_index != -1:
                labels = [[start_index], [start_index + len(a_token_ids) - 1]]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# Mask
class MaskedSoftmax(Layer):
    """
    在序列长度那一维进行softmax，并mask掉padding部分
    """
    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            inputs = inputs - (1.0 - mask) * 1e12
        return K.softmax(inputs, 1)

# 构建模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output = Dense(2)(model.output)
output = MaskedSoftmax()(output)
output = Permute((2, 1))(output)

model = Model(model.input, output)
model.summary()

# 评估函数
def sparse_categorical_crossentropy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[2])
    # 计算交叉熵
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def sparse_accuracy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    # 计算准确率
    y_pred = K.cast(K.argmax(y_pred, axis=2), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


# 编译模型
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(learing_rate),
    metrics=[sparse_accuracy]
)

# 答案抽取
def extract_answer(question, context, max_a_len=16):
    """
    抽取答案函数
    """
    max_q_len = 64
    q_token_ids = tokenizer.encode(question, maxlen=max_q_len)[0]
    c_token_ids = tokenizer.encode(
        context, maxlen=maxlen - len(q_token_ids) + 1
    )[0]
    token_ids = q_token_ids + c_token_ids[1:]
    segment_ids = [0] * len(q_token_ids) + [1] * (len(c_token_ids) - 1)
    c_tokens = tokenizer.tokenize(context)[1:-1]
    mapping = tokenizer.rematch(context, c_tokens)
    probas = model.predict([[token_ids], [segment_ids]])[0]
    probas = probas[:, len(q_token_ids):-1]
    start_end, score = None, -1
    for start, p_start in enumerate(probas[0]):
        for end, p_end in enumerate(probas[1]):
            if end >= start and end < start + max_a_len:
                if p_start * p_end > score:
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    return context[mapping[start][0]:mapping[end][-1] + 1]

# 预测文件生成
def predict_to_file(infile, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = {}
    for d in tqdm(load_data(infile)):
        a = extract_answer(d[2], d[1])
        R[d[0]] = a
    R = json.dumps(R, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()

# 官方评估函数
import sys
import io
import json
# sys.path.append('/content/drive/My Drive/kaikeba/project03/roberta/data')
from evaluate import evaluate as src_evaluate
from collections import OrderedDict

def evaluate(filename):
    """
    评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(filename, filename + '.pred.json')
    ref_ans = json.load(io.open(filename))
    pred_ans = json.load(io.open(filename + '.pred.json'))
    F1, EM, TOTAL, SKIP = src_evaluate(ref_ans, pred_ans)
    output_result = OrderedDict()
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    return output_result


class Evaluator(keras.callbacks.Callback):
    """
    评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(
            os.path.join(data_dir,'dev.json')
            # os.path.join(data_dir,'demo_dev.json')
        )
        if float(metrics['F1']) >= self.best_val_f1:
            self.best_val_f1 = float(metrics['F1'])
            model.save_weights(os.path.join(output_dir,'roberta_best_model.weights'))
            model.save(os.path.join(output_dir,'roberta_best_model.h5'))
        metrics['BEST_F1'] = self.best_val_f1
        print(metrics)

# 获取数据
train_generator = data_generator(train_data, batch_size)
evaluator = Evaluator()

# 模型训练

epochs=5
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    verbose=1,
    callbacks=[evaluator]
)


# 加载最优模型
from keras.models import load_model
model=load_model(os.path.join(output_dir,'roberta_best_model.h5'),custom_objects={'MaskedSoftmax':MaskedSoftmax,'sparse_accuracy':sparse_accuracy})
print(evaluate(os.path.join(data_dir,'dev.json')))

# 预测结果

# test1
predict_to_file(os.path.join(data_dir,'test1.json'), os.path.join(output_dir,'pred1.json'))

# test2
predict_to_file(os.path.join(data_dir,'test2.json'),  os.path.join(output_dir,'pred2.json'))

