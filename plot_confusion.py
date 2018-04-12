# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:33:19 2018

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


import re
regex = re.compile('\s+')
path = 'E:\\tfmodellogs\\evalresult\\val_confusion.txt'
f = open(path,'r')
cnf_matrix = np.zeros([26,26])
i = 0
for line in f.readlines():
    s = list(map(int,regex.split(line)[2:28]))
    cnf_matrix[i,:] = s
    i+=1


label_dict = {'market':1,'airplane':2,'swim':3,'fitting':4,'aquarium':5,'canteen':6,
              'rest room':7,'auto show':8,'plaza':9,'zoo':10,'ski':11,'park':12,
              'amu_p':13,'night':14,'moun':15,'sea':16,'sky':17,'snow':18,
              'island':19,'rainbow':20,'desert':21,'tree':22,'ancient building':23,'fountain':24,
              'vally':25}
    
class_names = list(label_dict.keys()) 
class_names.append('others')
## Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(15,15))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('E:\\tfmodellogs\\evalresult\\confusion.jpg')
plt.show()
