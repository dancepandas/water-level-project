# 加油
# 时间：2024/4/19 16:41
import numpy as np
import pandas as pd
df=pd.read_excel('D:\水科院\数据.xlsx',sheet_name='Sheet1')
df=df[df.columns[1]]
value=df.values.T
collect=[]
for i in range(value.shape[0]//30):
    t=value[30*i:30*(i+1)]
    sum=np.sum(t)
    collect.append(sum)

dfs=np.array(collect).reshape(len(collect),1)
dfss=pd.DataFrame(dfs)
dfss.to_excel('D:\水科院\数据1.xlsx')

