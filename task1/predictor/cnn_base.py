#encoding='utf-8'
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.utils import plot_model
from keras.layers import Dense,Dropout,Input,Flatten,Lambda,Activation
from keras.layers import Conv1D,MaxPooling1D,Embedding,BatchNormalization
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import fbeta_score,f1_score
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from  keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

max_sequence_length=175
num_words=20000
embedding_dim=300


def get_embed_metrix():
    print('get word vectors...')
    path='/home/wshong/PycharmProjects/CAIL2018/text_pre_process/processed_vector.txt'
    embeddings_matrix = [np.array([0 for x in range(embedding_dim)])]
    with open(path,'r')as fi:
        for line in fi.readlines():
            values = line.split()
            coefs = np.asarray(values, dtype='float32')
            embeddings_matrix.append(coefs)
    print('found %s word vectors.' % len(embeddings_matrix))
    return np.array(embeddings_matrix)

def get_data():
    print('get data...')
    train_path = '/home/wshong/Downloads/cail_0518/train/train_relation.txt'
    valid_path = '/home/wshong/Downloads/cail_0518/valid/valid_relation.txt'
    train_data=[]
    valid_data=[]
    with open(train_path, 'r')as fi:
        for line in fi.readlines():
            data = line.split()
            data= np.array(data,dtype='int32')
            train_data.append(data)
    with open(valid_path, 'r')as fi:
        for line in fi.readlines():
            data = line.split()
            valid_data.append(np.array(data,dtype='int32'))
    # train_data=np.array(train_data,dtype='int32')
    # valid_data=np.array(valid_data,dtype='int32')
    train_data = pad_sequences(train_data,max_sequence_length)
    valid_data=pad_sequences(valid_data,max_sequence_length)
    return train_data,valid_data
    #return pad_sequences(train_data,max_sequence_length),pad_sequences(valid_data,max_sequence_length)

def get_label(task):
    print('get label...')
    train_label=[]
    valid_label=[]
    train_path='/home/wshong/Downloads/cail_0518/train/train_'+task+'.txt'
    valid_path = '/home/wshong/Downloads/cail_0518/valid/valid_' + task + '.txt'

    with open(train_path,'r')as fi:
        for line in fi.readlines():
            label=line.strip()
            train_label.append(label)
    with open(valid_path,'r')as fi:
        for line in fi.readlines():
            label=line.strip()
            valid_label.append(label)

    train_label=to_categorical(np.asarray(train_label))
    valid_label = to_categorical(np.asarray(valid_label))

    return len(train_label[0]),train_label,valid_label

def get_class_weight(task):
    path='/home/wshong/Downloads/cail_0518/train/class_weight_'+task+'.txt'
    class_weight={}
    with open(path,'r')as fi:
        for i,weight in enumerate(fi.readlines()):
            class_weight[i] = float(weight.split()[0])
    return class_weight

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

def CNN_base(config):
    # weight加入glove预先训练好的向量，trainable设置成这部分参数不训练（glove预先训练好了，也可以添加自己的向量）
    embedding_layer = Embedding(config['num_words']+1, config['embedding_dim'], weights=[config['embed']], input_length=config['max_sequence_length'],
                                trainable=False)
    print('training model.')

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(300,3)(embedded_sequences)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling1D(3,padding='same')(x)
    x = Conv1D(300,3)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling1D(3,padding='same')(x)
    x = Conv1D(300,3)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling1D(3,padding='same')(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    #x = Dropout(0.25)(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    #x = Dropout(0.25)(x)
    preds = Dense(config['class_num'], activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.summary()
    return model

def train(config,task):
    """

    :param config:
    :param task: task 'accusation' for accusation classification,task 'law' for law classification, task 'time' for time...
    :return:
    """
    config['embed']= get_embed_metrix()
    config['train_data'],config['valid_data']= get_data()
    config['class_num'],config['train_label'],config['valid_label']=get_label(task)
    config['class_weight']=get_class_weight(task)


    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model=CNN_base(config)
    # f1_micro=fbeta_score
    # f1_macro = fbeta_score
    checkpoint=ModelCheckpoint('CNN_base_best2_'+task+'.h5',monitor='val_categorical_accuracy',mode='max',save_best_only=True,verbose=1)
    reduceLR=ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2, verbose=0, mode='max',
                                      epsilon=0.0001, cooldown=0, min_lr=0)
    earlystop=EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=4, mode='min')

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    model.fit(config['train_data'], config['train_label'], batch_size=128, epochs=50,verbose=1,callbacks=[checkpoint,reduceLR,earlystop],
              validation_data=(config['valid_data'], config['valid_label']), class_weight=config['class_weight'])
    #plot_model(model, to_file='cnn_1.png', show_shapes=True)
    #model.save('CNN_base_'+task+'.h5')

if __name__ == '__main__':
    config={}
    config['max_sequence_length'] = 175
    config['num_words']= 20000
    config['embedding_dim'] = 300
    tasks=['accusation','law','time']
    for task in tasks:
        train(config,task)

