#encodinf=utf8
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from sklearn import metrics
max_sequence_length=175
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
def format_result(result):
    rex = {"accusation": [], "articles": [], "imprisonment": -3}

    res_acc = []
    for x in result["accusation"]:
        if not (x is None):
            res_acc.append(int(x))
    rex["accusation"] = res_acc

    if not (result["imprisonment"] is None):
        rex["imprisonment"] = int(result["imprisonment"])
    else:
        rex["imprisonment"] = -3

    res_art = []
    for x in result["articles"]:
        if not (x is None):
            res_art.append(int(x))
    rex["articles"] = res_art

    return rex

def get_data():
    print('get data...')
    valid_path = '/home/wshong/Downloads/cail_0518/valid/valid_relation.txt'
    valid_data=[]
    with open(valid_path, 'r')as fi:
        for line in fi.readlines():
            data = line.split()
            valid_data.append(np.array(data,dtype='int32'))
    valid_data=pad_sequences(valid_data,max_sequence_length)
    return valid_data

def get_label(task):
    print('get label...')
    train_label=[]
    valid_label=[]
    valid_path = '/home/wshong/Downloads/cail_0518/valid/valid_' + task + '.txt'

    with open(valid_path,'r')as fi:
        for line in fi.readlines():
            label=line.strip()
            valid_label.append(int(label))

    #valid_label = to_categorical(np.asarray(valid_label))

    return np.asarray(valid_label)

if __name__ == '__main__':
    valid_data=get_data()
    task='time'
    valid_label=get_label(task)
    cnt = 0
    model=load_model('CNN_base_best_'+task+'.h5',custom_objects={'f1':f1})
    yp=model.predict(valid_data,batch_size=128,verbose=1)
    pred=np.argmax(yp,axis=1)

    print(metrics.classification_report(valid_label, pred,digits=4))


    # def get_batch():
    #     v = user.batch_size
    #     if not (type(v) is int) or v <= 0:
    #         raise NotImplementedError
    #
    #     return v
    #
    #
    # def solve(fact):
    #     result = user.predict(fact)
    #
    #     for a in range(0, len(result)):
    #         result[a] = format_result(result[a])
    #
    #     return result
    #
    #
    # for file_name in os.listdir(data_path):
    #     inf = open(os.path.join(data_path, file_name), "r")
    #     ouf = open(os.path.join(output_path, file_name), "w")
    #
    #     fact = []
    #
    #     for line in inf:
    #         fact.append(json.loads(line)["fact"])
    #         if len(fact) == get_batch():
    #             result = solve(fact)
    #             cnt += len(result)
    #             for x in result:
    #                 print(json.dumps(x), file=ouf)
    #             fact = []
    #
    #     if len(fact) != 0:
    #         result = solve(fact)
    #         cnt += len(result)
    #         for x in result:
    #             print(json.dumps(x), file=ouf)
    #         fact = []
    #     inf.close()
    #     ouf.close()
    #
    # jud = Judger(accusation_path='/home/wshong/Downloads/cail_0518/accu.txt',
    #              law_path='/home/wshong/Downloads/cail_0518/law.txt')
    # res = jud.test(truth_path=data_path, output_path=output_path)
    # jud.get_score(result=res)