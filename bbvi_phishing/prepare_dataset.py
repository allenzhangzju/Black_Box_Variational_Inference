import numpy as np
import csv

dim=68
train_labels=[]
train_features=[]
test_labels=[]
test_features=[]
dataNum=0
source='./dataset/phishing-train.txt'
'''
读取数据
'''
with open(source,'r') as f:
    datas=f.readlines()
    dataNum=len(datas)
'''
转换格式
'''
for i in range(dataNum):
    line=datas[i]
    position=line.find(' ')
    label=eval(line[:position])
    feature=eval('{'+line[position+1:len(line)-2].replace(' ',',')+'}')
    temp=np.zeros(dim)
    #temp[0]=1
    for key in feature.keys():
        temp[key-1]=1
    train_labels.append([(label-0.5)*2])
    train_features.append(temp)
train_labels=np.array(train_labels)
train_features=np.array(train_features)




f=open('./dataset/train_images_csv.csv','w',encoding='utf-8',newline='')
csv_writer=csv.writer(f)
csv_writer.writerows(train_features)
f.close()
f=open('./dataset/train_labels_csv.csv','w',encoding='utf-8',newline='')
csv_writer=csv.writer(f)
csv_writer.writerows(train_labels)
f.close()

