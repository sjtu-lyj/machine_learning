__author__ = 'church-father'
# my.oschina.net/bery/blog/203595

import numpy as np
import matplotlib.pyplot as plt

N=5
menMeans=(20,35,30,35,27)
menStd=(2,3,4,1,2)

ind=np.arange(N)
width=0.35

fig,ax=plt.subplots()
rects1=ax.bar(ind,menMeans,width,color='r',yerr=menStd)

womenMeans=(25,32,34,20,25)
womenStd=(3,5,2,3,3)
rects2=ax.bar(ind+width,womenMeans,width,color='y',yerr=womenStd)

ax.set_ylabel('Scoers')
ax.set_title('scort by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels(('01','02','03','04','05'))

ax.legend((rects1[0],rects2[0]),('man','women'))

def autolabel(rects):
    for rect in rects:
        height=rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2,1.05*height,'%d'%int(height),ha='center',va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
