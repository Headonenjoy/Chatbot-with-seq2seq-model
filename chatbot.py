import numpy as np 
import pandas as pd 
from pkuseg import pkuseg
import os
import yaml
import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import tensorflow as tf 
from tf.keras import activations,preprocessing,optimizers
from tf.keras.models import Model 
from tf.keras.layers import Dense,Input,LSTM,Dropout,Embedding

max_vocab_size=50000
embedding_dim=200
hidden_state_dim=200
epochs=150
batch_size=50
data_path='data'
stop_folder='stop_words'
stop_file='stop_words.txt'
outpath='output'

# 读取输入数据
def process_data(data_path):
    
    #dir_path = 'data'
    files_list = os.listdir(data_path + os.sep)

    questions = list()
    answers = list()

    for filepath in files_list:
        stream = open( data_path + os.sep + filepath , 'rb')
        docs = yaml.safe_load(stream)
        conversations = docs['conversations']
        for con in conversations:
            if len(con) > 2 :
                questions.extend(con[:-1])
                answers.extend(con[1:])
            elif len( con )> 1:
                questions.append(con[0])
                answers.append(con[1])
    questions.append('UNK')
    answers.append('UNK')
    answers_with_tags = list() #作为中间转换存在
    for i in range(len( answers )):
        if type(answers[i]) == str:
            answers_with_tags.append(answers[i]) #这跟answers一样
        else:
            questions.pop(i)

    answers = list()
    for i in range(len(answers_with_tags)) :
        answers.append('START ' + answers_with_tags[i] + ' END')
    
    return questions,answers

questions,answers=process_data(data_path)


# 读取停用词表
def stop_list(stop_file):
    stop_list=[]
    stop_path=os.sep.join([stop_folder,stop_file])
    with open(stop_path,'r',encoding='utf-8') as f:
        for stop_word in f:
            if stop_word:
                stop_list.append(stop_word.strip())
    return stop_list

class Tokenizer():
    def __init__(self):
        self.n=0
    def __call__(self,line):
        words=[word for word in pkuseg().cut(line)]
        return words

# 词频统计以及生成词典
count_vectorizer = CountVectorizer(tokenizer=Tokenizer(),max_features=50000,stop_words=stop_list(stop_file))
count_vectorizer.fit(questions + answers)    
vocabulary={key_:value_ for key_,value_ in count_vectorizer.vocabulary_.items()}
reverse_vocabulary={value_:key_ for key_,value_ in count_vectorizer.vocabulary_.items()}
vocab_size=len(vocabulary)
#joblib.dump(vocabulary,outpath+os.sep+'vocabulary.pkl')
#joblib.dump(reverse_vocabulary,outpath+os.sep+'reverse_vocabulary.pkl')
#joblib.dump(count_vectorizer,outpath+os.sep+'count_vectorizer.pkl')

analyzer=count_vectorizer.build_analyzer()
def words_to_indices(texts):
    word_indices=[]
    for text in texts:
        word_index=[vocabulary.get(token,vocabulary['unk']) for token in analyzer(text)]
        word_indices.append(word_index)
    return word_indices 


# encoder-decoder模型输入数据预处理
# encoder_input_data
tokenized_questions=words_to_indices(questions)
maxlen_questions=max([len(x) for x in tokenized_questions])
padded_questions=preprocessing.sequence.pad_sequences(tokenized_questions , maxlen=maxlen_questions , padding='post')
encoder_input_data=np.array(padded_questions)
print( encoder_input_data.shape , maxlen_questions) 

# decoder_input_data
tokenized_answers=words_to_indices(answers)
maxlen_answers=max([len(x) for x in tokenized_answers])
padded_answers=preprocessing.sequence.pad_sequences(tokenized_answers , maxlen=maxlen_answers , padding='post')
decoder_input_data=np.array(padded_answers)
print( (decoder_input_data.shape , maxlen_answers) )

# decoder_output_data
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = utils.to_categorical( padded_answers , vocab_size )
decoder_output_data = np.array( onehot_answers )
print( decoder_output_data.shape )


# 定义encoder-decoder模型
encoder_inputs = Input(shape=( None , ))
encoder_embedding = Embedding( vocab_size, 200 , mask_zero=True ) (encoder_inputs)
#参考链接：嵌入层 Embedding<https://keras.io/zh/layers/embeddings/#embedding>
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
#参考链接：https://keras.io/zh/layers/recurrent/#lstm
encoder_states = [state_h , state_c ]

decoder_inputs = Input(shape=( None , ))
decoder_embedding = Embedding( vocab_size, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = Dense( vocab_size , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=optimizers.RMSprop(), loss='categorical_crossentropy',metrics=['accuracy'])
#参考链接：RMSprop<https://keras.io/zh/optimizers/#rmsprop>
#categorical_crossentropy<https://keras.io/zh/backend/#categorical_crossentropy>

model.summary()

# 模型训练以及保存
model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150 ) 
model.save( 'model.h5' ) 

# 人机交互
def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model

def convert_questions(sentence):
    words_indices=[vocabulary.get(token,vocabulary['unk']) for token in analyzer(sentence)]
    return preprocessing.sequence.pad_sequences( [words_indices] , maxlen=maxlen_questions , padding='post')


enc_model , dec_model = make_inference_models()

for _ in range(10):
    states_values = enc_model.predict( convert_questions( input( '输入问题 : ' ) ) )
    empty_target_seq = np.zeros((1,1))
    empty_target_seq[0,0] = vocabulary['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict( [empty_target_seq]  + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in vocabulary.items() :
            if sampled_word_index == index :
                decoded_translation += '{}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    print( decoded_translation[:-3] )


