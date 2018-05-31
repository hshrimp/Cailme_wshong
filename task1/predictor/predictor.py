#encoding=utf8
from pyltp import Segmentor
from keras.models import load_model
import numpy as np
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
import re

def get_word_dict(path):
    word_dict = {}
    num = 1
    with open(path, 'r', encoding='utf-8')as fi:
        for line in fi.readlines():
            word_dict[line.strip()] = num
            num += 1
    return word_dict

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Predictor(object):
    def __init__(self):
        #self.tfidf = joblib.load('model/tfidf.model')
        self.law = load_model('CNN_base_best_law.h5',custom_objects={'f1':f1})
        self.accu = load_model('CNN_base_best_accusation.h5',custom_objects={'f1':f1})
        self.time = load_model('CNN_base_best_time.h5',custom_objects={'f1':f1})
        self.batch_size = 128
        self.max_sequence_length=175

        segmentor = Segmentor()  # 初始化实例,split words
        segmentor.load('/home/wshong/PycharmProjects/CAIL2018/text_pre_process/cws.model')  # 加载模型for cut text
        self.cut = segmentor.segment

        self.dict_path = '/home/wshong/PycharmProjects/CAIL2018/text_pre_process/word_dict.txt'
        self.word_dict=get_word_dict(self.dict_path)
        self.path = '/home/wshong/PycharmProjects/CAIL2018/text_pre_process/stopwords.txt'
        self.stopwords = []
        with open(self.path, 'r', encoding='utf-8')as fi:
            for line in fi.readlines():
                word = line.strip('\n')
                self.stopwords.append(word)
        self.pattern = re.compile(
            '([0-9]{4}年)([0-9][0-9]?月)?([0-9][0-9]?日)?(凌晨|上午|中午|下午|晚上|傍晚|晚|早上)?([0-9][0-9]?时)?([0-5]?[0-9]分)?(许|左右)?')

    def get_vec(self,content):
        vec = []
        if ('，' in content or '：' in content):
            content = re.split('，|：', content, 1)[1]  # remove the part before first('，|：'),like '公诉机关指控'
        content = re.sub(self.pattern, '', content)  # remove the date like '2016年3月3日18时许'
        words = self.cut(content)
        for word in words:
            i = self.word_dict.get(word)
            if i is None:
                continue
            else:
                vec.append(i)
        return vec

    def predict_law(self, vec):
        y = self.law.predict(vec,batch_size=self.batch_size)
        y = np.argmax(y, axis=1)
        #return [y[0] + 1]
        return [pred+1 for pred in y ]

    def predict_accu(self, vec):
        y = self.accu.predict(vec,batch_size=self.batch_size)
        y = np.argmax(y, axis=1)
        # return [y[0] + 1]
        return [pred + 1 for pred in y]

    def predict_time(self, vec):
        def ypred(y):
            # 返回每一个罪名区间的中位数
            if y == 0:
                return -2
            if y == 1:
                return -1
            if y == 2:
                return 120
            if y == 3:
                return 102
            if y == 4:
                return 72
            if y == 5:
                return 48
            if y == 6:
                return 30
            if y == 7:
                return 18
            else:
                return 6

        y = self.time.predict(vec,batch_size=self.batch_size)
        y = np.argmax(y, axis=1)
        return [ypred(yp) for yp in y]

    def predict(self, content):

        ans_list={}
        result=[]
        vec=[]
        for line in content:
            vec.append(self.get_vec(line))
        vec = pad_sequences(vec, self.max_sequence_length)

        ans_list['accusation'] = self.predict_accu(vec)
        ans_list['articles'] = self.predict_law(vec)
        ans_list['imprisonment'] = self.predict_time(vec)
        for num in range(0,len(ans_list['accusation'])):
            ans = {}
            ans['accusation']=[ans_list['accusation'][num]]
            ans['articles']=[ans_list['articles'][num]]
            ans['imprisonment']=ans_list['imprisonment'][num]

            print(ans)
            result.append(ans)
        return result
