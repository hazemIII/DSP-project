from scipy.stats._continuous_distns import triang_gen

__author__ = 'hazem'
import wave
import winsound
import os
from scipy.io.wavfile import read,write
from numpy.fft import fft , ifft
import numpy
import scipy
path=[]
for i in range (0,10):
    path.append("D:\\ELC4a\\DSP project\\speech_data\\"+str(i))
train_set=[]
test_set=[]
numb=[]
for k in range(0,10):
    dirs = os.listdir( path[k] )
    for i in range(0,20):
        rate,data=read(path[k]+"/"+dirs[i])
        numb.append(data)
    train_set.append(numb)
    numb=[]
    for i in range(20,30):
        rate,data=read(path[k]+"/"+dirs[i])
        numb.append(data)
    test_set.append(numb)
    numb=[]
#s=wave.open('1.wav','rb')
#winsound.PlaySound('test.wav',winsound.SND_FILENAME)

################################Filter Bank #################################################
F_train=[]
F_test = []

for i in train_set:
    for k in i:
        numb.append(fft(k))
    F_train.append(numb)
    numb=[]
for i in test_set:
    for k in i:
        numb.append(fft(k))
    F_test.append(numb)
    numb=[]
l1= [300 ,500]
l2 = [500 ,800]
l3=[800 ,1200]
l4=[1200 ,1800]
l5=[1800,3000]
l6=[3000,4500]
l7=[4500,7000]
FF_train = []
filtered=[]

for i in F_train:
    for k in i :

        f1 = numpy.zeros(len(k))
        f2 = numpy.zeros(len(k))
        f3 = numpy.zeros(len(k))
        f4 = numpy.zeros(len(k))
        f5 = numpy.zeros(len(k))
        f6 = numpy.zeros(len(k))
        f7 = numpy.zeros(len(k))
        f1[l1[0]:l1[1]] = 1
        f2[l2[0]:l2[1]] = 1
        f3[l3[0]:l3[1]] = 1
        f4[l4[0]:l4[1]] = 1
        f5[l5[0]:l5[1]] = 1
        f6[l6[0]:l6[1]] = 1
        f7[l7[0]:l7[1]] = 1
        filtered.append(f1*k)
        filtered.append(f2*k)
        filtered.append(f3*k)
        filtered.append(f4*k)
        filtered.append(f5*k)
        filtered.append(f6*k)
        filtered.append(f7*k)
        numb.append(filtered)
        filtered=[]
    FF_train.append(numb)
    numb=[]
N=[]
for i in FF_train:
    for k in i:
        numb.append(abs(ifft(k)))
    N.append(numb)
    numb=[]
knowledge=[]
temp=[]
sum=numpy.zeros(7)
sum1=0
for i in N:
    for k in i:
        for m in k:
            sum1=numpy.log(numpy.sum(numpy.power(m,2)))
            temp.append(sum1)
        sum=sum+numpy.asarray(temp)
        temp=[]
    sum=sum/20
    knowledge.append(sum)
    sum=numpy.zeros(7)


print(knowledge)
print(knowledge[0])
print(len(knowledge))
scipy.io.savemat('knowledge.mat', {'knowledge':knowledge})


