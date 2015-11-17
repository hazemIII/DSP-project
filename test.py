__author__ = 'hazem'
import scipy
import numpy
from scipy.io.wavfile import read,write
from numpy.fft import fft , ifft
import os , wave
from operator import itemgetter
test_set=[]
numb=[]
path = []
for i in range (0,10):
    path.append("D:\\ELC4a\\DSP project\\speech_data\\"+str(i))
for k in range(0,10):
    dirs = os.listdir( path[k] )
    for i in range(20,30):
        rate,data=read(path[k]+"/"+dirs[i])
        numb.append(data)
    test_set.append(numb)
    numb=[]
knowledge=scipy.io.loadmat('knowledge.mat')

def feature(wave):
    Fwave= fft(wave)
    l1= [300 ,500]
    l2 = [500 ,800]
    l3=[800 ,1200]
    l4=[1200 ,1800]
    l5=[1800,3000]
    l6=[3000,4500]
    l7=[4500,7000]
    f1 = numpy.zeros(len(wave))
    f2 = numpy.zeros(len(wave))
    f3 = numpy.zeros(len(wave))
    f4 = numpy.zeros(len(wave))
    f5 = numpy.zeros(len(wave))
    f6 = numpy.zeros(len(wave))
    f7 = numpy.zeros(len(wave))
    f1[l1[0]:l1[1]] = 1
    f2[l2[0]:l2[1]] = 1
    f3[l3[0]:l3[1]] = 1
    f4[l4[0]:l4[1]] = 1
    f5[l5[0]:l5[1]] = 1
    f6[l6[0]:l6[1]] = 1
    f7[l7[0]:l7[1]] = 1
    n=[]
    n.append(abs(ifft(f1*wave)))
    n.append(abs(ifft(f2*wave)))
    n.append(abs(ifft(f3*wave)))
    n.append(abs(ifft(f4*wave)))
    n.append(abs(ifft(f5*wave)))
    n.append(abs(ifft(f6*wave)))
    n.append(abs(ifft(f7*wave)))
    feature_list=[]
    for i in n:
        feature_list.append(numpy.log(numpy.sum(numpy.power(i,2))))
    return feature_list
def classifier(F,K):
    s = []
    for i in range(0,10):
        s.append(numpy.sum(abs((F - K[i]))))
    index = min(enumerate(s), key=itemgetter(1))
    return index[0]

W = wave.open('1.wav','rb')
rate,data=read('1.wav')
counter = 0
c_f=numpy.zeros([10,10])
for i in range(0,10):
    for n in range(0,10):
        f = feature(test_set[i][n])
        c= classifier(f,knowledge["knowledge"])
        c_f[i][c]=c_f[i][c]+1
        if i == c :
            counter=counter+1
print(counter)
print(c_f)
