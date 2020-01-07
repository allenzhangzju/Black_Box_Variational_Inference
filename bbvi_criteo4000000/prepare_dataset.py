import numpy as np

dim=1000000
train_labels=[]
train_features=[]
test_labels=[]
test_features=[]
dataNum=0
source='./dataset/criteo-train-sub500.txt'
'''
读取数据
'''
with open(source,'r') as f:
    datas=f.readlines()
    dataNum=len(datas)
datas=np.array(datas)[np.array([5,7])]
'''
转换格式
'''
for i in range(2):
    line=datas[i]
    position=line.find(' ')
    label=eval(line[:position])
    feature=eval('{'+line[position+1:len(line)-2].replace(' ',',')+'}')
    temp=np.zeros(dim)
    #temp[0]=1
    for key in feature.keys():
        temp[key]=0.16013
    train_labels.append([(label-0.5)*2])
    train_features.append(temp)
train_labels=np.array(train_labels)
train_features=np.array(train_features)

train_data=np.column_stack((train_labels,train_features))

test=1
