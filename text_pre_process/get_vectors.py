#coding='utf-8'
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_nb_words=20000
embedding_dim=300
max_sequence_length=175 #found average length about 175 in valid data

def index_word_vectors():
    print('indexing word vectors.')

    embeddings_index={}
    path = '/home/wshong/Downloads/cail_0518/vector/vectors.txt'
    with open(path, 'r', encoding='utf-8')as f:
        for line in f:
            values=line.split()
            word=values[0]
            coefs=np.asarray(values[1:],dtype='float32')
            embeddings_index[word]=coefs

    print('found %s word vectors.' %len(embeddings_index))
    return embeddings_index

def get_text(data_type):
    """
    get the text from processed text file by text type, text type include train, test and valid
    :param data_type:
    :return:
    """
    print('getting the ',data_type,' text...')
    path='/home/wshong/Downloads/cail_0518/'+data_type+'/'+data_type+'_fact_processed.txt'
    text=[]
    with open(path,'r',encoding='utf-8')as fi:
        for line in fi.readlines():
            text.append(line.strip())
    return text

def create_relation_file(data,data_type):
    """
    create the relation file,
    the number in param seq is the word's place point in the word dictionary,
    :param seq:
    :return:
    """
    print('create '+data_type+' relation file...')
    path = '/home/wshong/Downloads/cail_0518/' + data_type + '/' + data_type + '_relation.txt'
    fi = open(path, 'w')
    fi.writelines('\n'.join(' '.join(str(num) for num in text_seq) for text_seq in data))
    fi.close()


def to_tensor(text_train,text_test,text_valid):
    """
    tokenizer and sequence
    :param text_train:
    :param text_test:
    :param text_valid:
    :return: word_index:like a dictionary but sort by words frequency, index from 1 to length of words,
                the most frequency word's index is 1.
    """
    tokenizer=Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(text_train+text_test+text_valid)

    sequences_train = tokenizer.texts_to_sequences(text_train)
    sequences_test = tokenizer.texts_to_sequences(text_test)
    sequences_valid = tokenizer.texts_to_sequences(text_valid)

    data_train = pad_sequences(sequences_train, max_sequence_length)
    data_test = pad_sequences(sequences_test, max_sequence_length)
    data_valid = pad_sequences(sequences_valid, max_sequence_length)

    word_index=tokenizer.word_index
    print('found %s unique tokens.'%len(word_index))
    return word_index,data_train,data_test,data_valid

def prepare_embedding(word_index,embeddings_index):
    print('preparing embedding matrix.')
    num_words=min(max_nb_words,len(word_index))
    embedding_matrix=np.zeros((num_words+1,embedding_dim))

    for word,i in word_index.items():
        if (i>=max_nb_words):
            continue
        embedding_vector=embeddings_index.get(word)
        if (embedding_vector is not None):
            embedding_matrix[i]=embedding_vector

    return num_words , embedding_matrix

def create_words_dict(word_index):
    print('creating word dictionary..')
    index = {i: word for word, i in word_index.items()}
    for i in range(1, max_nb_words + 1):
        word_dict.append(index[i])
    fi = open('word_dict.txt', 'w')
    fi.writelines('\n'.join(word_dict))
    fi.close()

def create_processed_vector(embedding_matrix):
    print('creating processed vector...')
    fi = open('processed_vector.txt', 'w')
    fi.writelines('\n'.join(' '.join(str(num) for num in vector) for vector in embedding_matrix[1:]))
    fi.close()

if __name__ == '__main__':
    text_train=get_text('train')
    text_test = get_text('test')
    text_valid = get_text('valid')
    word_dict=[]

    embeddings_index=index_word_vectors()
    word_index, data_train, data_test, data_valid=to_tensor(text_train,text_test,text_valid)

    #create the relation file
    create_relation_file(data_train, 'train')
    create_relation_file(data_test, 'test')
    create_relation_file(data_valid, 'valid')

    num_words, embedding_matrix=prepare_embedding(word_index,embeddings_index)

    create_processed_vector(embedding_matrix)

    create_words_dict(word_index)


