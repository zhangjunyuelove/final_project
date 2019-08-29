import os
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np

from tensorflow import gfile
b=os.listdir(r'/home/junyue/25grouptf')
c=[]
for row in b:
    file_path=os.path.join('/home/junyue/25grouptf', row)
    c.append(file_path)

c=np.array(c).astype(str)
kf = KFold(n_splits=25)
for train_index, test_index in kf.split(c):
    X_train, X_test = c[train_index], c[test_index]
    X_train_real = ','.join(X_train)
    
    




files = gfile.Glob('/home/junyue/yt8m/v2/frame/train*.tfrecord')
files2=gfile.Glob(X_train)

print(files2)
print(files)
