#!/usr/bin/env python
# coding: utf-8

# ## 加载库 超参数

# In[2]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import random
input_attributes = '微博中文内容'
output_attributes = '情感倾向'
DATA_PATH = '/data/wwwwww931121/2019COVID_weibo/'
BERT_PATH = '/data/wwwwww931121/bert_base_chinese/'
MAX_SEQUENCE_LENGTH = 140


# ## 加载数据集

# In[3]:


df_train = pd.read_csv(DATA_PATH+'nCoV_100k_train.labled.csv', encoding='utf-8', usecols=[3,6])
df_test = pd.read_csv(DATA_PATH+'nCov_10k_test.csv', encoding='utf-8', usecols=[0,3])
df_train = df_train[df_train[output_attributes].isin(['0','-1','1'])]
#print(df_train[output_attributes].value_counts())
df_sub = pd.read_csv(DATA_PATH+'submit_example.csv', encoding='utf-8')


# ## 加载预训练数据 分词

# In[4]:


def compute_bert_inputs(text, tokenizer, max_sequence_length):
    
    inputs = tokenizer.encode_plus(text,max_length=max_sequence_length)
    input_ids = inputs['input_ids']
    input_masks = [1] * len(input_ids)
    input_segments = inputs['token_type_ids']
    padding_length = max_sequence_length - len(input_ids)
    padding_id = tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)
    
    return [input_ids, input_masks, input_segments]

def compute_bert_input_arrays(df, col, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df[col]):
        ids, masks, segments = compute_bert_inputs(str(instance), tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)
           ]


# In[5]:


tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-chinese-vocab.txt')
inputs = compute_bert_input_arrays(df_train, input_attributes, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_bert_input_arrays(df_test, input_attributes, tokenizer, MAX_SEQUENCE_LENGTH)


# In[6]:


def compute_bert_output_arrays(df, col):
    return np.asarray(df[col].astype(int)+1)
outputs = compute_bert_output_arrays(df_train, output_attributes)


# ## 构建模型

# In[7]:


def create_model():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)    
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)    
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)    
    config = BertConfig.from_pretrained(BERT_PATH + 'bert-base-chinese-config.json') 
    config.output_hidden_states = False 
    bert_model = TFBertModel.from_pretrained(
        BERT_PATH+'bert-base-chinese-tf_model.h5', config=config)
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    embedding = bert_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    
    x = tf.keras.layers.GlobalAveragePooling1D()(embedding)    
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x)
    
    return model


# In[8]:


#print(outputs[0:100])


# In[9]:


def generate_data(inputs, outputs, trainnb, batch_size, shuffle=True):
    while True:
        c = [ i for i in range(int(trainnb/batch_size))]
        if shuffle:
            random.shuffle(c)
        x_b = []
        y_b = []
        for i in c:
            x1=np.array(inputs)[0][i*batch_size:(i+1)*batch_size]
            x2=np.array(inputs)[1][i*batch_size:(i+1)*batch_size]
            x3=np.array(inputs)[2][i*batch_size:(i+1)*batch_size]
            x_b = [x1,x2,x3]
            y_b = [outputs[i*batch_size:(i+1)*batch_size]]
            yield(x_b,y_b)


# In[ ]:


gkf = StratifiedKFold(n_splits=6).split(X=df_train[input_attributes].fillna('-1'), y=df_train[output_attributes].fillna('-1'))

valid_preds = []
test_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = to_categorical(outputs[train_idx])

        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = to_categorical(outputs[valid_idx])
        
        K.clear_session()
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
        model.fit(train_inputs, train_outputs, validation_data= [valid_inputs, valid_outputs], epochs=2, batch_size=32)
        # model.save_weights(f'bert-{fold}.h5')
        valid_preds.append(model.predict(valid_inputs))
        test_preds.append(model.predict(test_inputs))


# ## 输出

# In[13]:


sub = np.average(test_preds, axis=0)
sub = np.argmax(sub,axis=1)
df_sub['y'] = sub-1
df_sub['id'] = df_sub['id'].apply(lambda x: str(x)+' ')
df_sub.to_csv('/output/submit8.csv',index=False, encoding='utf-8')


# In[14]:


'''
dict={"测试数据id":df2['微博id'].values.tolist(),'情感极性':pred_label.tolist()}
#output_list = [df2['微博id'].values.tolist(),pred_label.tolist()]
output = pd.DataFrame(dict)
output.to_csv("",sep=',',index=False)
'''


# In[ ]:




