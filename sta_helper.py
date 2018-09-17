#coding:UTF-8
import sys
# from DATA_orgin.data_helper import *
import numpy as np
import math
import random as rd

import matplotlib.pyplot as plt

#绘制model分数和人工分数的散点图
#1.4102444016788234
def mean_square_error(s1,s2):
    sum = 0
    for idx, score in enumerate(s1):
        sum += math.pow(score - s2[idx], 2)
    return math.pow(sum / len(s1), 0.5)

def baseline(human_score):
    human_s = human_score.copy()
    rd_=np.array([rd.randint(0,4) for i in range(len(human_s))])
    mean_=np.array([np.mean(human_s) for i in range(len(human_s))])
    return mean_square_error(human_s,rd_),mean_square_error(human_s,mean_)

def baseline_auc(human_score):
    human_s=human_score.copy()
    rd_ = np.array([rd.random() for i in range(len(human_s))])
    mean_ = np.array([np.mean(human_s) for i in range(len(human_s))])
    return Auc(human_s,rd_),Auc(human_s,mean_)

def normal_toint(predict_s,yuzhi=0.5):
    out=[]
    for i in range(len(predict_s)):
        out.append(max(min(int(predict_s[i]+1-yuzhi),4),0))
    return np.array(out)

def score_int(predict_s,human_s,yuzhi=0.5):
    predict_s, human_s = predict_s.copy(), human_s.copy()
    predict_s=normal_toint(predict_s,yuzhi=yuzhi)
    n=0
    for i,j in zip(predict_s,human_s):
        if i==j:
            n+=1
    return n/len(predict_s)

def sort(human_s,pre_s):
    if len(human_s)<=1:
        return human_s,pre_s
    else:
        i=0
        j=len(human_s)-1
        rd1=human_s[rd.randint(0,len(human_s)-1)]
        flag=True
        while i<j:
            flag=not flag
            while rd1>human_s[i]:
                i+=1
            while rd1<human_s[j]:
                j-=1
            if i>=j:
                continue
            if human_s[i]==human_s[j]:
                if flag:
                    i+=1
                else:
                    j-=1
                continue
            else:
                tmp=human_s[i]
                human_s[i]=human_s[j]
                human_s[j]=tmp

                tmp=pre_s[i]
                pre_s[i]=pre_s[j]
                pre_s[j]=tmp

        human_s[:j],pre_s[:j]=sort(human_s[:j],pre_s[:j])
        human_s[j+1:],pre_s[j+1:]=sort(human_s[j+1:],pre_s[j+1:])
        return human_s,pre_s


def Auc(human_score,predict_score,flag=False):
    predict_s, human_s = predict_score.copy(), human_score.copy()
    predict_s,human_s=sort(predict_s, human_s)
    # if flag:
    #     looup(predict_s, human_s, True)
    idx_o=0
    idx=0

    while idx<len(predict_s):
        if idx==0:
            idx+=1
            continue
        else:
            while idx<len(predict_s) and predict_s[idx]==predict_s[idx-1]:
                idx+=1
            if idx-idx_o>=2:
                human_s[idx_o:idx], predict_s[idx_o:idx] = sort(human_s[idx_o:idx], predict_s[idx_o:idx])
            idx_o=idx
            idx+=1

    pos=np.sum(human_s)
    neg=len(human_s)-pos
    auc=0
    x_o=0
    y_o=0
    min_=1
    yuzhi=0
    for i in range(len(human_s)-1,-1,-1):
        x=(len(human_s[i:])-np.sum(human_s[i:]))/neg
        y=np.sum(human_s[i:])/pos
        if flag:
            plt.plot([x,x_o],[y,y_o],'b')
        if abs(-x+1-y)<min_:
            min_=abs(-x+1-y)
            yuzhi=predict_s[i]
        auc+=((y_o+y)*(x-x_o))/2
        y_o=y
        x_o=x
    plt.show()
    return auc,yuzhi

def looup(human_score,predict_score,all=False):
    print('\n',end='')
    k=0
    for i,j in zip(human_score,predict_score):
        if not all:
            if k>50:
                break
            k+=1
        print(i,j)

def normal(predict_s):
    i = 0
    scale = 10
    while (i - 1) * scale < len(predict_s):
        kk = np.mean(predict_s[i * scale:min(len(predict_s), (i + 1) * scale)])
        predict_s[i * scale:min(len(predict_s), (i + 1) * scale)] = kk
        i += 1
    return predict_s


def recall_and_pre(human_s,predict_s,yuzhi=0.5,num=5):
    tmp=normal_toint(predict_s,yuzhi)
    out=np.zeros((num,num))
    for i,j in zip(human_s,tmp):
        out[i,j]+=1
    for i in out:
        for j in i:
            print(int(j),end=' ')
        print()
    for idx in range(num):
        print(idx,'recall:',out[idx,idx]/np.sum(out[idx,:]),'pre:',(out[idx,idx]/np.sum(out[:,idx]))if np.sum(out[:,idx])!=0 else 0)

def resolve_filename(filename):
    dic={}
    filename=[each.split('_') for each in filename.replace('_ckpt','').split('/')[-1].split('-')]
    for each in filename:
        key='_'.join(each[:-1])
        if key=='LR':
            dic[key]=float(each[-1])
        elif key=='weight':
            dic[key] = each[-1] == str(True)
        elif key=='normal':
            dic[key]=each[-1] == str(True)
        else:
            dic[key]=each[-1]
    return dic


if __name__=='__main__':
    predict_s=np.load('./model/predict_score/best_predict_score.npy')

    dp=data_helper.data_helper()
    human_score = dp.get_old_test_data()

    baseline_rd,baseline_mean=baseline(human_score)

    print('baseline_rd:',baseline_rd)
    print('baseline_mean:',baseline_mean)

    print('mean_square_error',mean_square_error(predict_s,human_score))
    print('pre:',score_int(predict_s,human_score))
    print(human_score.tolist())

    #排序&&归一
    human_score,predict_s=sort(human_score,predict_s)
    predict_s=normal(predict_s)

    # x=np.array([i for i in range(len(human_score))])
    # y1=human_score
    # y2=predict_s
    #
    # # plt.scatter(x,y1,s=2)
    # # plt.scatter(x,y2,s=2)
    # plt.scatter(y2,y1,s=2)
    # plt.xlabel('model_score')
    # plt.ylabel('human_score')
    #
    # plt.show()





