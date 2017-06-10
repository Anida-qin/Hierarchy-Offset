
import numpy as np

import pylab as pl

f1=open('precision_recall_56','r')
lines1 = f1.readlines()
x1=[]
y1=[]
for line1 in lines1:
    a,b = line1.split(',')
    aa=a.split('(')[1]
    bb=b.split(')')[0]
    y1.append(float(aa))
    x1.append(float(bb))

plot1 = pl.plot(x1, y1,'g',label='epoch-50')# u
pl.title('Precision recall for different epoch')# give plot a title

pl.xlabel('recall')# make axis labels

pl.ylabel('precision')

pl.xlim(0.0, 0.7)# set axis limits

pl.ylim(0.0, 1.0)
pl.show()# show the plot on the screen
