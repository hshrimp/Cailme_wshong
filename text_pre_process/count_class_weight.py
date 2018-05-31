#encoding=utf8
import numpy as np
from sklearn.utils import class_weight


def read_label(path):
    labels=[]
    with open(path,'r')as fi:
        for label in fi.readlines():
            labels.append(float(label))
    return labels


if __name__ == '__main__':
    task='law'
    task_class={'time':9,'accusation':202,'law':183}
    path='/home/wshong/Downloads/cail_0518/train/train_'+task+'.txt'
    labels=read_label(path)
    ave=len(labels)/task_class[task]
    weight=[]
    la={}
    for i in range(0, task_class[task]):
        count = sum(i == lab for lab in labels)
        la[i] = count
        if(count ==0):
            weight.append('1.0')
        else:
            weight.append(str(ave/la[i]))
    # weight = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(labels),
    #                                                   labels)

    with open('/home/wshong/Downloads/cail_0518/train/class_weight_'+task+'.txt','w')as fi:
        fi.writelines('\n'.join(weight))












