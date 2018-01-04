# -*- coding: utf-8 -*-
import tensorflow.contrib.keras as kr
import numpy as np
from collections import Counter
import os
import json
from nltk.tokenize import word_tokenize


data_path="../data/our_data"
funny_dir=os.path.join(data_path,'funny.json')
unfunny_dir=os.path.join(data_path,'nofunny.json')
vocab_dir=os.path.join(data_path,'vocab.txt')
train_dir=os.path.join(data_path,'train.txt')
val_dir=os.path.join(data_path,'val.txt')
test_dir=os.path.join(data_path,'test.txt')


def read_file(contents,labels,filename,label):
    with open(filename,'r',encoding='utf-8',errors='ignore') as f:
        data=json.load(f)
        for item in data:
            text=item['text'].lower()
            text=word_tokenize(text)
            text = ['"' if x == '``' else x for x in text]
            contents.append(text)
            labels.append(label)


def split_data(funny_file,unfunny_file):
    contents,labels=[],[]
    read_file(contents,labels,funny_file,1)
    read_file(contents,labels,unfunny_file,0)
    contents=np.asarray(contents)
    labels=np.asarray(labels)
    indices=np.random.permutation(np.arange(len(contents)))
    contents_shuffle=list(contents[indices])
    labels_shuffle=list(labels[indices])
    index=int(len(contents)*0.8)
    index2=int(len(contents)*0.9)
    train_contents=contents_shuffle[:index]
    train_labels=labels_shuffle[:index]
    val_contents=contents_shuffle[index:index2]
    val_labels=labels_shuffle[index:index2]
    test_contents=contents_shuffle[index2:]
    test_labels=labels_shuffle[index2:]
    # print(len(contents_shuffle),len(train_contents),len(val_contents),len(test_contents))
    return train_contents,train_labels,val_contents,val_labels,test_contents,test_labels

def write_data(contents,labels,file):
    with open(file,'w',encoding='utf-8',errors='ignore') as f:
        for i in range(len(contents)):
            f.write(str(labels[i])+'\t'+json.dumps(contents[i])+'\n')
# train_contents,train_labels,val_contents,val_labels,test_contents,test_labels=split_data(funny_dir,unfunny_dir)
# write_data(train_contents,train_labels,os.path.join(data_path,'train.txt'))
# write_data(val_contents,val_labels,os.path.join(data_path,'val.txt'))
# write_data(test_contents,test_labels,os.path.join(data_path,'test.txt'))

#get data
def read_data(file):
    with open(file,'r',encoding='utf-8',errors='ignore') as f:
        contents,labels=[],[]
        lines=f.readlines()
        for line in lines:
            # print(line.split('\t'))
            label,content=line.split('\t')
            content=content.replace('\n','')
            labels.append(int(label))
            contents.append(json.loads(content))
        return contents,labels

# train_contents,_,_,_,_,_=get_data(funny_dir,unfunny_dir)
# for i in train_contents:
#     print(i)

# 取前5000词频占0.9351339526181685
def build_vocab(vocab_dir,vocab_size=5000):
    train_data,_,_,_,_,_=split_data(funny_dir,unfunny_dir)
    words=[]
    for content in train_data:
        words.extend(content)
    counter=Counter(words)
    total=sum(counter.values())
    counter_pairs=counter.most_common(vocab_size-2)
    words,values=list(zip(*counter_pairs))
    choosed_value=sum(values)
    print(choosed_value*1.0/total)
    words=['<PAD>']+['UNK']+list(words)
    with open(vocab_dir,'w') as f:
        for word in words:
            try:
                f.write(word+'\n')
            except:
                pass
# build_vocab(vocab_dir)

def read_vocab(vocab_dir):
    words=open(vocab_dir,'r',encoding='utf-8',errors='ignore').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def to_id(data,word_to_id):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] in word_to_id:
                data[i][j]=word_to_id[data[i][j]]
            else:
                data[i][j]=word_to_id['UNK']
    return data


def to_words(content,words):
    return ''.join(words[x] for x in content)

def file_to_id(word_to_id,content):
    data=to_id(content,word_to_id)
    return data

def get_data(file,word_to_id,seq_length,num_classes):
    inputs,labels=read_data(file)
    inputs=file_to_id(word_to_id,inputs)
    inputs=kr.preprocessing.sequence.pad_sequences(inputs,seq_length)
    labels=kr.utils.to_categorical(labels,num_classes)
    return inputs,labels

def batch_iter(x,y,batch_size):
    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        if end_id-start_id<batch_size:
            break
        yield x[start_id:end_id],y[start_id:end_id]



# _,word_to_id=read_vocab(vocab_dir)
#
# train_dir=os.path.join(data_path,'train.txt')
# train_inputs,train_labels=get_data(train_dir,word_to_id)
# train_batch=batch_iter(train_inputs,train_labels,batch_size,num_classes)
# for content,label in train_batch:
#     print(label)
#     print(content)
#     print('---------------')
