# 加油
# 时间：2023/9/20 13:01
import copy
import dataprocess as dp
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os

class PadMaskLayer(tf.keras.layers.Layer):
    def __init__(self, constant_values=-1):
        super(PadMaskLayer, self).__init__()
        self.constant_values = constant_values

    @tf.function
    def call(self, pad_data):
        mask = tf.not_equal(pad_data,self.constant_values)
        mask = tf.cast(mask, tf.float32)
        return mask
    def get_config(self):
        config = {
            'constant_values': self.constant_values
        }
        base_config = super(PadMaskLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ContaLayer(tf.keras.layers.Layer):
    def __init__(self,num_heads ,bl_dim,key_dim,rate,constant_values):
        super(ContaLayer, self).__init__()
        self.att = keras.layers.MultiHeadAttention(  num_heads=num_heads, key_dim=key_dim)
        self.blstm1 = keras.layers.Bidirectional(keras.layers.LSTM(bl_dim, return_sequences=True))
        self.blstm2 = keras.layers.Bidirectional(keras.layers.LSTM(bl_dim, return_sequences=True))
        self.norm = keras.layers.LayerNormalization(epsilon=1e-8)
        self.dropout = tf.keras.layers.Dropout(rate)
    @tf.function
    def call(self, inp, mask,training):
        att_output = self.att(inp, inp,inp, attention_mask=mask)
        att_output = self.dropout(att_output, training=training)
        out1 = self.norm(inp + att_output)
        blstm1_output = self.blstm1(out1)
        blstm1_outpu2 = self.blstm2(blstm1_output)
        return blstm1_outpu2

class AssigmentLayer(tf.keras.layers.Layer):
    def __init__(self,max_len,input_dim,constant_values,**kwargs):
        '''
        :param max_len: 最大序列长度
        :param input_dim: 特征维度
        :param constant_values: 填充值
        :param kwargs: 其他参数
        '''
        super(AssigmentLayer, self).__init__(**kwargs)
        self.max_len=max_len
        self.input_dim=input_dim
        self.constant_values=constant_values
    def build(self,input_shape):
            self.w1 = self.add_weight(name='w1',
                                     shape=((1,self.max_len)),
                                     initializer='random_normal',
                                     trainable=True)
            self.w2 = self.add_weight(name='w2',
                                      shape=((1, self.max_len)),
                                      initializer='random_normal',
                                      trainable=True)
            super(AssigmentLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        cut_inp1=inputs[:,:1,:1]
        cut_inp2=inputs[:,:1,1:2]
        result1= tf.matmul(cut_inp1, self.w1)
        result2 = tf.matmul(cut_inp2, self.w2)
        result1=tf.transpose(result1,perm=[0,2,1])
        result2 = tf.transpose(result2, perm=[0, 2, 1])
        result_con_01=tf.concat([result1,result2],axis=2)
        result_con_01=tf.reshape(result_con_01,shape=(-1,2))
        tens=result_con_01
        for i in range(1,(self.max_len-inputs.shape[2])//2):
            tens1=tf.fill((i,2),self.constant_values)
            tens1=tf.cast(tens1,tf.float32)
            tens2=tf.concat([tens1,result_con_01[:-i]],axis=0)
            tens=tf.concat([tens,tens2],axis=1)
        tens_split=tf.split(tens, num_or_size_splits=30, axis=0)
        tens=tf.stack(tens_split,axis=1)
        tensor_data=tf.concat([tens,inputs[:,:,2:]],axis=2)
        return tensor_data

class PaddingLayer(tf.keras.layers.Layer):
    def __init__(self, max_len, constant_values):
        super(PaddingLayer, self).__init__()
        self.max_len = max_len
        self.constant_values = constant_values

    @tf.function
    def call(self, input_data):
        pad_len1 = self.max_len - input_data.shape[1]
        pad_len2 = self.max_len - input_data.shape[2]
        paddings = tf.convert_to_tensor([[0, 0], [0, pad_len1], [0, pad_len2]])
        padded_data = tf.pad(input_data, paddings, constant_values=self.constant_values)
        return padded_data

class Model_part(tf.keras.Model):
    def __init__(
        self,  max_len,input_dim,num_heads ,ff_dim,bl_dim,key_dim,d_model,constant_values,m,rate=0.1):
        '''
        :param max_len: 最大序列长度
        :param input_dim:特征维度
        :param num_heads: 多头注意力机制的头数
        :param ff_dim: 前馈层神经元个数
        :param bl_dim: 双向层神经元个数
        :param key_dim: 多头注意力机制键数
        :param d_model: 输出结果维度
        :param rate: 随机丢失率
        :param m conta层层数
        :param constant_values:填充值
        '''
        super(Model_part,self).__init__()
        self.padding = PaddingLayer(max_len,constant_values)
        self.padmask = PadMaskLayer(constant_values)
        self.assigment=AssigmentLayer(max_len,input_dim,constant_values)
        self.ffn = keras.Sequential(
            [ keras.layers.Dense(ff_dim, activation="tanh"),
                keras.layers.Dense(max_len, activation="tanh")] )
        self.conta_layer = ContaLayer(num_heads, bl_dim, key_dim, rate,constant_values)
        self.ff_final = tf.keras.layers.Dense(d_model, activation="tanh")
        self.dropout = tf.keras.layers.Dropout(rate)
        self.m=m
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    @tf.function
    def call(self, inputs,training):
        assg_out01=self.assigment(inputs)
        inpt=self.padding(assg_out01)
        mask = self.padmask(inpt)
        inpt=self.norm(inpt)
        conta_out = self.conta_layer(inpt,mask)
        for i in range(self.m):
            conta_out=self.conta_layer(conta_out,mask)
        ffn_output=self.ffn(conta_out)
        ffn_output=self.dropout(ffn_output,training=training)
        final_out=self.ff_final(ffn_output)
        return final_out

class Firstmodel():
    def __init__(self,config_file,inputs):
        '''
        :param config_file: 文件路径
        '''
        self.config_file = config_file
        self.inputs=inputs
        self.load_config()
    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.max_len = config["max_len"]
        self.num_heads = config["num_heads"]
        self.ff_dim= config["ff_dim"]
        self.bl_dim = config["bl_dim"]
        self.key_dim = config["key_dim"]
        self.d_model = config["d_model"]
        self.m = config["m"]
        self.constant_values=config["constant_values"]
        self.loss=config['loss']
        self.metrics=config['metrics']

    def creat_model(self):
        model=Model_part(self.max_len,self.inputs['input_dim'], self.num_heads, self.ff_dim, self.bl_dim, self.key_dim, self.d_model, self.m,self.constant_values)
        return model

    def training(self,model,epochs,batch_size,learning_rate):
        inp=self.inputs['train_data']
        c=self.inputs['train_result']
        inp_test=self.inputs['test_data']
        test_result=self.inputs['test_result']
        def lr_scheduler(epoch, lr):
            if epoch %200==0 and epoch!=0:
                c=0.1
                return lr*c
            else:
                return lr

        class EvaluateCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                loss, accuracy = model.evaluate(inp_test, test_result, verbose=0)
                print(f'-loss:{loss:.5f}  -accuracy:{accuracy:.5f}')
        evaluate_callback = EvaluateCallback()
        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        early_stopping_callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=self.loss,metrics=[self.metrics])
        model.fit(inp, c, epochs=epochs, batch_size=batch_size,callbacks=[lr_scheduler_callback,evaluate_callback,early_stopping_callback],validation_data=(inp_test,test_result))



    @staticmethod
    def save(model,path):
        model.save_weights(path)
        file_path_dir = os.path.dirname(os.path.realpath(__file__))
        print("模型保存完成！保存路径:",os.path.join(file_path_dir, path))

def model_predict(save_path,config_file,inputs):
    '''
    :param save_path: 模型保存路径
    :param config_file: 模型初始参数文件
    :param inputs:模型输入
    :return: 模型
    '''
    xyyt=Firstmodel(config_file,inputs)
    model=xyyt.creat_model()
    model.load_weights(save_path)
    test_data=inputs['test_data']
    result=model.predict(test_data)
    return result



if __name__ == '__main__':

    #保存外部输入参数
    def save_config(config_file):
        file_path_dir = os.path.dirname(os.path.realpath(__file__))
        config = {
            "max_len": 30,
            "num_heads": 8,
            "ff_dim": 128,
            "bl_dim": 128,
            "key_dim": 2,
            "d_model": 1,
            "m": 1,
            "constant_values":-1,
            'loss':'MAE',
            'metrics':'MSE'
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print("参数保存完成！保存路径:", os.path.join(file_path_dir, config_file))
    '''
    config_file为模型输入参数文件路径
    '''
    path='D:\水科院\数据.xlsx'
    config_file="config_file"
    save_config(config_file)
    inputs=dp.pad([path,'Sheet4'],[path,'Sheet3'],[path,'Sheet5'],config_file)
    xyt=Firstmodel(config_file,inputs)
    model=xyt.creat_model()
    '''
    epochs为模型训练迭代次数
     batch_size为模型训练一次使用的数据量
      learning_rate为模型学习率
    '''
    epochs=500
    batch_size=1
    learning_rate=5e-7
    xyt.training(model,epochs,batch_size=batch_size,learning_rate=learning_rate)
    '''
    save_path为模型权重配置文件存储路径
    '''
    save_path="haizigongyuan.index"
    config_file = "config_file"
    xyt.save(model,save_path)
    print("模型保存完成")
    y=model_predict(save_path,config_file,inputs)
    df=pd.DataFrame(y[:,:,0])
    df.to_excel("D:\水科院\数据1.xlsx",sheet_name="Sheet3",index=True)
    #=OFFSET($A$1,ROW(A30)/30-1,MOD(ROW(A30),30))&""



