from __future__ import print_function
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from random import randint, seed
from skimage.transform import rotate
from keras.layers import Merge
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras import callbacks
import gzip
import sys
from six.moves import cPickle
from keras.regularizers import l2, activity_l2
import os
import ntpath
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure
from skimage import io, exposure, img_as_uint, img_as_float
import os
import gzip
import sys
from keras.utils.np_utils import to_categorical
from six.moves import cPickle
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
import cv2
from skimage.transform import swirl
from skimage.transform import resize
import numpy
# Lets make sure we can reproduce our results
seed( 1024 )
np.random.seed(1337)  # for reproducibility
K.set_image_dim_ordering('th')
ntpath.basename("a/b/c")
dir="./isolated_char.pkl.gz"
#dir = sys.argv[1]
path="./iso_char_cnn_weight/"
if not os.path.exists(path):
	os.makedirs(path)
print (path)
	#print ("Pixel Per Cell:%d" %ppc)
	#print(len(X_train))

def load_data( path ):
    
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')
    
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding="bytes")
    
        f.close()
        return data  #(data, labels)
    




dirname="./ISI-BW-NRSZ-CR-TH_1.pkl.gz"

img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = load_data( dirname )
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#X_val=X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255



i=0
train_data_hfd=[]
train_label=[]
test_data_hdf=[]
test_label=[]
train_data_reg=[]
test_data_reg=[]
x=1
epoch_num=0

#print (path)
#print ("Pixel Per Cell:%d" %ppc)
print("Training Data is Getting Ready!")


#Convert Float 32

#train_data_reg = train_data_reg.astype('float32')
#X_test = X_test.astype('float32')
#X_val = X_val.astype('float32')
#train_data_reg /= 255
#X_test /= 255
#X_val /= 255


batch_size = 128
nb_classes = 10
nb_epoch = 100


nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


# convert class vectors to binary class matrices
#HoG
train_label = np_utils.to_categorical(train_label, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Reguler Dataset
#Y_test = np_utils.to_categorical(y_test, nb_classes)
#Y_val=np_utils.to_categorical(y_val, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Dropout(0.50))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.50))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Dropout(0.50))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Dropout(0.50))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

model.summary()
mod_wgt_dir="./EE3+.hdf5"
print (mod_wgt_dir)
model.load_weights(mod_wgt_dir)
pred=model.predict_classes(X_test, batch_size=100, verbose=1)
print(len(pred))
score = model.evaluate(X_test, Y_test, verbose=0)

print(score)

confusion_matrix = numpy.zeros( (10, 10 ))

for k in range( len(X_test) ):
	#print(k)
	#print("prediction",pred[i])
	#print("actual",test_label[i])
  	confusion_matrix[ y_test[k] ] [pred[k]] = confusion_matrix[ y_test[k] ] [pred[k]] +1
print (confusion_matrix)
print(type(confusion_matrix))
import seaborn as sn
import seaborn as sn
import pandas as pd
import numpy as np
array = confusion_matrix.astype(int)
df_cm = pd.DataFrame(array, index = [i for i in "0123456789"],
                  columns = [i for i in range(10)])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt="d",cmap="GnBu",vmin=380, vmax=400)
plt.savefig('confusion_matrix.png', format='png')

