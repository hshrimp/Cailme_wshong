import re
from pyltp import Segmentor
import json
from text_pre_process import data

class Process():
    def __init__(self):
        self.path='stopwords.txt'
        self.stopwords=[]
        with open(self.path,'r',encoding='utf-8')as fi:
            for line in fi.readlines():
                word=line.strip('\n')
                self.stopwords.append(word)
        segmentor = Segmentor()  # 初始化实例,split words
        segmentor.load('cws.model')  # 加载模型
        self.cut = segmentor.segment
        #self.pattern = re.compile('([0-9]{4}年)(((0?[13578]|1[02])月(0?[1-9]|[12][0-9]|3[01])日)|(([469]|11)月([1-9]|[12][0-9]|30)日)|(0?2月([1-9]|[1][0-9]|2[0-9])日))(凌晨|上午|中午|下午|晚上|傍晚|晚|早上)?([0-9][0-9]?时)?([0-5]?[0-9]分)?(许|左右)?')
        self.pattern = re.compile(
        '([0-9]{4}年)([0-9][0-9]?月)?([0-9][0-9]?日)?(凌晨|上午|中午|下午|晚上|傍晚|晚|早上)?([0-9][0-9]?时)?([0-5]?[0-9]分)?(许|左右)?')

    def main_fun(self,content):
        if('，' in content or'：' in content):
            content=re.split('，|：', content, 1)[1] #remove the part before first('，|：'),like '公诉机关指控'
        content=re.sub(self.pattern,'',content) #remove the date like '2016年3月3日18时许'
        words=self.cut(content)
        #print(' '.join(words))
        words_new=[]
        for word in words:
            word=str(word).strip()
            if(word not in self.stopwords):
                words_new.append(word)
        print(' '.join(words_new))
        return words_new


def read_trainData(path):
    fin = open(path, 'r', encoding='utf8')

    alltext = []

    accu_label = []
    law_label = []
    time_label = []

    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])
        accu_label.append(data.getlabel(d, 'accu'))
        law_label.append(data.getlabel(d, 'law'))
        time_label.append(data.getlabel(d, 'time'))
        line = fin.readline()
    fin.close()

    return alltext, accu_label, law_label, time_label
if __name__ == '__main__':
    # path = '/home/wshong/PycharmProjects/CAIL2018/text_pre_process/stopwords.txt'
    #
    # str = '公诉机关指控，2016年3月3日18时许，中华人民共和国刑法被告人易某某行至达州市通川区大观园公交车站附近，扒窃了被害人黎某某的裤袋内一部白色三星S某型手机，在逃离现场时被公安民警当场抓获。经达州市通川区价格认证中心鉴定，该手机价值人民币1450元，现已发还给被害人黎某某。'
    # a = re.split('，|：', str, 1)
    # cont = []
    # with open(path, 'r')as fi:
    #     for line in fi.readlines():
    #         word = line.strip('\n')
    #         cont.append(line.strip('\n'))
    #         if (word in '中华人民共和国刑法ss.'):
    #             print('aaaaa')
    # print(cont)
    # if ('中华人民共和国刑法' in cont):
    #     print('ssss')
    # segmentor = Segmentor()  # 初始化实例
    # segmentor.load('/home/wshong/PycharmProjects/CAIL2018/text_pre_process/cws.model')  # 加载模型
    # words = segmentor.segment(str)
    # print(' '.join(words))
    cutted_text=[] #split to words ,and concatenate by space to a string
    cutted_words=[] #split to words,words sequences
    print('reading...')
    data_type='test'
    alltext, accu_label, law_label, time_label = read_trainData('/home/wshong/Downloads/cail_0518/data_'+data_type+'.json')
    print('cut text...')
    pro = Process()
    for line in alltext:
        words=pro.main_fun(line)
        cutted_words.append(words)
        cutted_text.append(' '.join(words)+'\n')
    f=open('/home/wshong/Downloads/cail_0518/'+data_type+'/'+data_type+'_fact_processed.txt','w')
    f.writelines(cutted_text)
    f.close()
    f=open('/home/wshong/Downloads/cail_0518/'+data_type+'/'+data_type+'_time.txt','w')
    f.writelines('\n'.join(str(time) for time in time_label))
    f.close()
    f = open('/home/wshong/Downloads/cail_0518/' + data_type + '/' + data_type + '_accusation.txt', 'w')
    f.writelines('\n'.join(str(time) for time in accu_label))
    f.close()
    f = open('/home/wshong/Downloads/cail_0518/' + data_type + '/' + data_type + '_law.txt', 'w')
    f.writelines('\n'.join(str(time) for time in law_label))
    f.close()
