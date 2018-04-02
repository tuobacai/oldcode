# coding:utf-8
'''
数据处理函数
应用于Siamese LSTM的data util
输入文本为清洗好的文本,格式为
seq1_token1 seq1_token2 seq1_token2 ... seq1_tokenN\tseq2_token1 seq2_token2 seq2_token3 ... seq2_tokenN\tlabel
文本1与文本2以及label用"`"隔开
文本之间的token使用空格" "隔开
label为0或1表示相似与不相似
'''
import os
import cPickle
import numpy as np
from collections import defaultdict
class DataUtils(object):
    def __init__(self, config, is_train=True):
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length

        vocab_file = os.path.join(config.data_dir, 'vocab.pkl')
        input_file = os.path.join(config.data_dir, config.train_file)

        if not (os.path.exists(vocab_file)):
            print 'readling train file'
            self.preprocess(input_file, vocab_file)
        else:
            print 'loading vocab file'
            self.load_vocab(vocab_file)

        if is_train:
            self.create_batches(input_file)
            self.reset_batch()
        else:
            self.handle_data(input_file)

    '''得到词汇表'''
    def  preprocess(self, input_file, vocab_file, min_freq=2):
        token_freq=defaultdict(int)
        for line in open(input_file):
            label,seq1,seq2=line.rstrip().split('`')
            seq=seq1+' '+seq2
            for token in seq.split(' '):
                token_freq[token]+=1

        token_list = [w for w in token_freq.keys() if token_freq[w] >= min_freq]
        token_list.append('<pad>')
        token_dict = {token: index for index, token in enumerate(token_list)}

        with open(vocab_file, 'w') as f:
            cPickle.dump(token_dict, f)
        self.token_dictionary=token_dict
        self.vocab_size=len(self.token_dictionary)

    '''输入数据处理'''
    def load_vocab(self, vocab_file):

        with open(vocab_file, 'rb') as f:
            self.token_dictionary = cPickle.load(f)
            self.vocab_size = len(self.token_dictionary)

    '''文本数组转成单词编号数据'''
    def text_to_array(self, text, is_clip=True):

        seq_ids = [int(self.token_dictionary.get(token)) for token in text if
                   self.token_dictionary.get(token) is not None]
        if is_clip:
            seq_ids = seq_ids[:self.sequence_length]
        return seq_ids

    '''padding处理'''
    def padding_seq(self, seq_array, padding_index):

        for i in xrange(len(seq_array), self.sequence_length):
            seq_array.append(padding_index)

    '''创建批量数据'''
    def create_batches(self, text_file):
        x1=[]
        x2=[]
        y=[]

        padding_index=self.vocab_size-1
        for line in open(text_file,'r'):
            label,seq1, seq2 = line.rstrip().split('`')
            seq1_array = self.text_to_array(seq1.split(' '))
            seq2_array = self.text_to_array(seq2.split(' '))

            self.padding_seq(seq1_array, padding_index)
            self.padding_seq(seq2_array, padding_index)

            label = int(label)
            x1.append(seq1_array)
            x2.append(seq2_array)
            y.append(label)

        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)
        self.num_samples=len(y)
        self.num_batches = self.num_samples / self.batch_size
        indices = np.random.permutation(self.num_samples)
        self.x1 = x1[indices]
        self.x2 = x2[indices]
        self.y = y[indices]

    '''预处理数据'''
    def handle_data(self, text_file):
        x1 = []
        x2 = []
        y = []

        padding_index = self.vocab_size - 1
        for line in open(text_file, 'r'):
            label, seq1, seq2 = line.rstrip().split('`')
            seq1_array = self.text_to_array(seq1.split(' '))
            seq2_array = self.text_to_array(seq2.split(' '))

            self.padding_seq(seq1_array, padding_index)
            self.padding_seq(seq2_array, padding_index)

            label = int(label)
            x1.append(seq1_array)
            x2.append(seq2_array)
            y.append(label)

        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)
        self.num_samples = len(y)
        indices = np.random.permutation(self.num_samples)
        self.x1 = x1[indices]
        self.x2 = x2[indices]
        self.y = y[indices]

    '''重置批量数据'''
    def reset_batch(self):
        self.pointer = 0
        self.eos = False

    '''获取批量数据'''
    def next_batch(self):
        begin=self.pointer
        end=self.pointer+self.batch_size
        x1_batch=self.x1[begin:end]
        x2_batch = self.x2[begin:end]
        y_batch = self.y[begin:end]

        new_pointer=self.pointer+self.batch_size
        if new_pointer>=self.num_samples:
            self.eos=True
        else:
            self.pointer=new_pointer
        return x1_batch,x2_batch,y_batch










