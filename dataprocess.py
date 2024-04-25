# 加油
# 时间：2024/4/19 9:11
import copy

import pandas as pd
import json
import numpy as np

#文件路径(仅接受xlsx文件和csv文件)
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

#读取开采量和补水量数据
def read_exploitation(path,sheet_name='Sheet1'):
    '''
    :param path: 文件路径
    :param sheet_name: excel文件sheet名字,csv文件不用填写
    :return:
    '''
    target=path[-3:]
    if target=='lsx':
        df=pd.read_excel(path,sheet_name=sheet_name)
        values=df.values
        return values
    elif target=='csv':
        df=pd.read_csv(path)
        values = df.values
        return values

#读取其他特征数据
def read_others(path,sheet_name='Sheet1'):
    '''
    :param path: 文件路径
    :param sheet_name: 文件sheet名字,csv文件不用填写
    :return:
    '''
    target = path[-3:]
    if target == 'lsx':
        df = pd.read_excel(path, sheet_name=sheet_name)
        values = df.values
        return values
    elif target == 'csv':
        df = pd.read_csv(path)
        values = df.values().T
        return values

def read_result(path,sheet_name='Sheet1'):
    '''
    :param path: 文件路径
    :param sheet_name: 文件sheet名字,csv文件不用填写
    :return:
    '''
    target = path[-3:]
    if target == 'lsx':
        df = pd.read_excel(path, sheet_name=sheet_name)
        values = df.values.tolist()
        return values
    elif target == 'csv':
        df = pd.read_csv(path)
        values = df.values.tolist()
        return values

#数据处理
def pad(lst1,lst2,lst3,config_file):
    config=load_config(config_file)
    max_len=config['max_len']
    constant_values=config['constant_values']
    exp_hyd=read_exploitation(lst1[0],sheet_name=lst1[1])
    other_value=read_others(lst2[0],sheet_name=lst2[1])
    result_value=read_result(lst3[0],sheet_name=lst3[1])
    input_dim=other_value.shape[-1]
    te = copy.copy(other_value)
    for m in range(1,max_len//other_value.shape[1]):
        te=np.concatenate([te,np.concatenate([np.full((m,other_value.shape[1]),constant_values),other_value[:-m]],axis=0)],axis=1)
    cut_num = max_len // (2 * other_value.shape[1])
    te = te[:, :-cut_num * 2]
    other_value=te
    if other_value.shape[0]%max_len!=0:
        print('数据维度错误！数据总体长度与预测尺度【max_len】不是倍数关系')
    else:
        collect=[]
        collect_result=[]
        for i in range(other_value.shape[0]//max_len):
            oe=other_value[max_len*i:max_len*(i+1)]
            result_e=result_value[max_len*i:max_len*(i+1)]
            ee=np.zeros((max_len,2))
            ee[0:,]=exp_hyd[i:,][0]
            oe=np.concatenate([ee,oe],axis=1)
            collect.append(oe)
            collect_result.append(result_e)
        values=np.array(collect)
        result_values=np.array(collect_result)
        train_data=values[:int(values.shape[0]*0.8)]
        train_result=result_values[:int(values.shape[0]*0.8)]
        test_data=values[int(values.shape[0]*0.8):]
        test_result = result_values[int(values.shape[0]*0.8):]
        return {'train_data':train_data,'train_result':train_result, 'test_data': test_data,'test_result':test_result,'input_dim':input_dim}






