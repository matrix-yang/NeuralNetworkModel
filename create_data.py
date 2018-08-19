#code:utf-8
import numpy as np
import matplotlib.pyplot as plt

seed=2

def generateds():
    #基于seed产生随机数
    rdm=np.random.RandomState(seed)
    #随机数返回300行2列的矩阵，表示300组坐标点（x0,x1）
    #rdm.randn是一种产生标准正态分布的随机数或矩阵的函数
    X=rdm.randn(300,2)
    #print(X)
    #半径为根号2的圆 ，圆内为大点1 ，圆外为点0
    Y_=[int(x1*x1+x0*x0<2) for(x0,x1) in X]
    #print(Y_)
    #圆内为红色，圆外为蓝色
    Y_c=[ 'red'if c else 'blue'for c in Y_]

    Y_=np.vstack(Y_).reshape(-1,1)
    return X,Y_,Y_c

