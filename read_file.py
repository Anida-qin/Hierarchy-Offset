import numpy as np

import pylab as pl





f=open('/home/qs/Double_Duel_results/15epoch/59-15epoch/precision_recall_56','r')
lines = f.readlines()
x=[]
y=[]
for line in lines:
    a,b = line.split(',')
    aa=a.split('(')[1]
    bb=b.split(')')[0]
    y.append(float(aa))
    x.append(float(bb))

pl.plot(x, y)# use pylab to plot x and y

pl.title('Plot of y vs. x')# give plot a title

pl.xlabel('x axis')# make axis labels

pl.ylabel('y axis')

pl.xlim(0.0, 0.4)# set axis limits

pl.ylim(0.0, 1.0)
pl.show()# show the plot on the screen